# hashdir

import hashlib
import os
import sys


def hash_directory(path, *, kind='md5'):
    ''' Walk the directories and files under path in deterministic order
        and compute a hash of contents.
        __pycache__ directories are skipped.
    '''
    use = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        }[kind]
    digest = use()

    for root, dirs, files in sortedWalk(
        path, skip_dir=lambda x: x in ['__pycache__', '.git', 'proc', '.ipynb_checkpoints'],
    ):
        for names in files:
            if names == '.gitignore':
                continue
            file_path = os.path.join(root, names)
            if file_path.endswith(".swp"):
                continue

            # Hash the path and add to the digest to account for empty
            # files and directories.
            digest.update(use(file_path[len(path):].encode()).digest())

            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f_obj:
                    while True:
                        buf = f_obj.read(1024 * 1024)
                        if not buf:
                            break
                        digest.update(buf)

    return digest.hexdigest()


def sortedWalk(top, *, topdown=True, onerror=None, skip_dir=lambda x: False):
    ''' Walk into directories in filesystem.
        Ripped from os module and slightly modified
        for alphabetical sorting (os.walk is not deterministic).
    '''
 
    names = os.listdir(top)
    names.sort()
    dirs, nondirs = [], []
 
    for name in names:
        if os.path.isdir(os.path.join(top, name)):
            if skip_dir(name):
                continue
            dirs.append(name)
        else:
            nondirs.append(name)
 
    if topdown:
        yield top, dirs, nondirs
    for name in dirs:
        path = os.path.join(top, name)
        if not os.path.islink(path):
            for x in sortedWalk(
                path, topdown=topdown, onerror=onerror, skip_dir=skip_dir,
            ):
                yield x
    if not topdown:
        yield top, dirs, nondirs



if __name__ == "__main__":
    print(hash_directory(sys.argv[1]))
