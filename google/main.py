"""Drivers to do some common stuff with google backups"""

import re

from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from tarfile import TarFile

from nkpylib.google.constants import BACKUPS_DIR

def get_tar_files(backups_dir: str, user: str, force: bool=False, **kw) -> list[Path]:
    """Get a list of tar files in the backups directory for a specific user."""
    backups_dir = Path(backups_dir)
    if not backups_dir.exists():
        raise FileNotFoundError(f"Backups directory '{backups_dir}' does not exist.")
    pattern = f"{user}*.tgz"
    tar_files = list(backups_dir.glob(pattern))
    if not tar_files:
        print(f"No tar files found for user '{user}' in '{backups_dir}'.")
    return tar_files

def extract_re(pattern: str, **kw):
    """Extract files matching a regexp pattern from tar files in the backups directory."""
    pat = re.compile(pattern)
    print(f'Extracting files matching pattern: {pattern}')
    tar_files = get_tar_files(**kw)
    print(f'Got {len(tar_files)} tar files to extract from: {tar_files}')
    for tar_file in tar_files:
        print(f'Extracting from tar file: {tar_file}')
        with TarFile.open(tar_file, 'r:gz') as tar:
            for member in tar.getmembers():
                if pat.search(member.name):
                    print(f'Extracting {member.name} from {tar_file}')
                    output_path = Path(kw['backups_dir']) / member.name
                    if output_path.exists() and not kw.get('force', False):
                        continue
                    if not output_path.parent.exists():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                    tar.extract(member, path=kw['backups_dir'])


if __name__ == '__main__':
    fns = {fn.__name__: fn for fn in [extract_re]}
    parser = ArgumentParser(description='Google backups driver')
    parser.add_argument('fn', choices=fns.keys(), help='Function to execute')
    parser.add_argument('pattern', type=str, help='Regular expression pattern to match files')
    parser.add_argument('--backups-dir', default=BACKUPS_DIR, help='Dir where backups are stored')
    parser.add_argument('--user', default='apu', help='User to use for backups (prefix-matching)')
    parser.add_argument('--force', action='store_true', help='Force extraction even if files already exist')
    args = parser.parse_args()
    kwargs = vars(args)
    fn = fns[kwargs.pop('fn')]
    fn(**kwargs)
