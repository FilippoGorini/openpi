import collections.abc
import concurrent.futures
import datetime
import fnmatch
import logging
import os
import pathlib
import re
import shutil
import stat
import subprocess
import time
import urllib.parse

import filelock
import fsspec
import fsspec.generic
import tqdm_loggable.auto as tqdm

# Environment variable to control cache directory path, ~/.cache/openpi will be used by default.
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
DEFAULT_CACHE_DIR = "~/.cache/openpi"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(
    url: str,
    *,
    force_download: bool = False,
    ignore_patterns: collections.abc.Sequence[str] | None = None,
    **kwargs,
) -> pathlib.Path:
    """Download a file or directory from a remote filesystem to the local cache, and return the local path.

    If the local file already exists, it will be returned directly.

    It is safe to call this function concurrently from multiple processes.
    See `get_cache_dir` for more details on the cache directory.

    Args:
        url: URL to the file to download.
        force_download: If True, the file will be downloaded even if it already exists in the cache.
        ignore_patterns: Optional glob patterns (matched against each file's path relative to `url`) that
            will be skipped when downloading a directory, e.g. `("train_state/*",)` to avoid pulling the
            optimizer state that is not needed for inference. Ignored for single-file downloads.
        **kwargs: Additional arguments to pass to fsspec.

    Returns:
        Local path to the downloaded file or directory. That path is guaranteed to exist and is absolute.
    """
    # Don't use fsspec to parse the url to avoid unnecessary connection to the remote filesystem.
    parsed = urllib.parse.urlparse(url)

    # Short circuit if this is a local path.
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()

    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    # Check if the cache should be invalidated.
    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path

    try:
        lock_path = local_path.with_suffix(".lock")
        with filelock.FileLock(lock_path):
            # Ensure consistent permissions for the lock file.
            _ensure_permissions(lock_path)
            # First, remove the existing cache if it is expired.
            if invalidate_cache:
                logger.info(f"Removing expired cached entry: {local_path}")
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()

            if not local_path.exists():
                # Download the data to a local cache.
                logger.info(f"Downloading {url} to {local_path}")
                scratch_path = local_path.with_suffix(".partial")
                # Route openpi-assets through gsutil to avoid gcsfs auth issues with this bucket.
                # All other gs:// URLs (e.g. big_vision) continue to use gcsfs as normal.
                # gsutil cp has no per-file exclude, so fall back to the fsspec path when the caller
                # asked to skip some files.
                if parsed.scheme == "gs" and parsed.netloc == "openpi-assets" and not ignore_patterns:
                    _download_gsutil(url, scratch_path, **kwargs)
                else:
                    _download_fsspec(url, scratch_path, ignore_patterns=ignore_patterns, **kwargs)

                shutil.move(scratch_path, local_path)
                _ensure_permissions(local_path)

    except PermissionError as e:
        msg = (
            f"Local file permission error was encountered while downloading {url}. "
            f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
        )
        raise PermissionError(msg) from e

    return local_path


def _download_gsutil(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """Download a file or directory from GCS using gsutil if available, otherwise fall back to gcsfs."""
    if shutil.which("gsutil") is None:
        logger.warning(
            "gsutil not found, falling back to gcsfs. This may fail if GCP credentials are not configured correctly."
        )
        _download_fsspec(url, local_path, **kwargs)
        return
    local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{url}/*", str(local_path)],
        check=True,
    )


def _download_fsspec(
    url: str,
    local_path: pathlib.Path,
    *,
    ignore_patterns: collections.abc.Sequence[str] | None = None,
    **kwargs,
) -> None:
    """Download a file from a remote filesystem to the local cache, and return the local path."""
    fs, root = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    # Folders are represented by 0-byte objects with a trailing forward slash.
    is_dir = info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))

    # When filtering a directory, enumerate the files ourselves and fetch only the ones we keep,
    # so ignored subtrees (e.g. train_state/) are never transferred over the network.
    if is_dir and ignore_patterns:
        rpaths, lpaths, total_size = _plan_filtered_get(fs, url, root, local_path, ignore_patterns)
        if not rpaths:
            local_path.mkdir(parents=True, exist_ok=True)
            return
        for lp in lpaths:
            pathlib.Path(lp).parent.mkdir(parents=True, exist_ok=True)
        get_args = (rpaths, lpaths)
        get_kwargs = {}
    else:
        total_size = fs.du(url) if is_dir else info["size"]
        # Pass a str target: some fsspec backends (e.g. HfFileSystem for hf://) reject a
        # pathlib.Path local target and raise inside the worker thread.
        get_args = (url, str(local_path))
        get_kwargs = {"recursive": is_dir}

    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, *get_args, **get_kwargs)
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        # Surface any exception from the worker thread; otherwise a failed download is
        # swallowed here and only shows up later as a misleading FileNotFoundError on move.
        future.result()
        pbar.update(total_size - pbar.n)


def _plan_filtered_get(
    fs, url: str, root: str, local_path: pathlib.Path, ignore_patterns: collections.abc.Sequence[str]
) -> tuple[list[str], list[str], int]:
    """List the files under `url`, drop those matching `ignore_patterns`, and return the paired
    (remote paths, local paths, total size in bytes) for the files that should be downloaded.

    Patterns are matched with `fnmatch` against each file's path relative to `url` (POSIX separators),
    so `*` also matches `/` (e.g. `train_state/*` skips the whole train_state subtree).
    """
    # detail=True avoids a size() round-trip per file for the progress bar.
    detail = fs.find(url, withdirs=False, detail=True)
    root_prefix = root.rstrip("/") + "/"

    rpaths: list[str] = []
    lpaths: list[str] = []
    total_size = 0
    for remote_path, entry in detail.items():
        rel = remote_path[len(root_prefix) :] if remote_path.startswith(root_prefix) else remote_path.lstrip("/")
        if any(fnmatch.fnmatch(rel, pat) for pat in ignore_patterns):
            continue
        rpaths.append(remote_path)
        lpaths.append(str(local_path / rel))
        total_size += entry.get("size") or 0
    return rpaths, lpaths, total_size


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod requires executable permission to be set, so we skip if the permission is already match with the target."""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """Set folder permission to be read, write and searchable."""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """Since we are sharing cache directory with containerized runtime as well as training script, we need to
    ensure that the cache directory has the correct permissions.
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """Set all files to be read & writable, if it is a script, keep it as a script."""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _get_mtime(year: int, month: int, day: int) -> float:
    """Get the mtime of a given date at midnight UTC."""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


# Map of relative paths, defined as regular expressions, to expiration timestamps (mtime format).
# Partial matching will be used from top to bottom and the first match will be chosen.
# Cached entries will be retained only if they are newer than the expiration timestamp.
_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """Invalidate the cache if it is expired. Return True if the cache was invalidated."""

    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            # Remove if not newer than the expiration timestamp.
            return local_path.stat().st_mtime <= expire_time

    return False
