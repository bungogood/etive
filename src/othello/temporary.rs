use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::path::Path;

use tempfile::Builder;

pub(super) fn atomic_file_save(
    path: &Path,
    save: impl FnOnce(&Path) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let temporary = Builder::new()
        .prefix(".etive-")
        .suffix(&temporary_suffix(path))
        .tempfile_in(parent)?;
    save(temporary.path())?;
    temporary.persist(path).map_err(|error| error.error)?;
    Ok(())
}

fn temporary_suffix(path: &Path) -> OsString {
    let extension = path.extension().unwrap_or_else(|| OsStr::new("tmp"));
    let mut suffix = OsString::from(".");
    suffix.push(extension);
    suffix
}
