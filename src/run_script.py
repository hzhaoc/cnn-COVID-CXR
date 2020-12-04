if __name__ == "__main__":

    import pathlib
    import shutil
    import subprocess
    import sys
    import os
    import time
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # https://github.com/tensorflow/tensorflow/issues/37649


    _, notebook, timeout = sys.argv

    notebook = pathlib.Path(notebook)
    timeout = int(timeout)

    if notebook.name.split('.')[-1] == 'ipynb':
        # From full pathname, copy the notebook file to the current directory.
        shutil.copyfile(notebook, notebook.name)

        # Run it in place as a notebook.
        subprocess.run([
            'jupyter',
            'nbconvert',
            '--ExecutePreprocessor.timeout={}'.format(timeout),
            '--to', 'notebook',
            '--inplace',
            '--execute', notebook.name,
        ])

        # Convert what just ran to html
        subprocess.run([
            'jupyter',
            'nbconvert',
            '--to', 'html',
            notebook.name,
        ])

        # delete notebooks after run completion
        time.sleep(2)
        os.remove(notebook.name)

    elif notebook.name.split('.')[-1] == 'py':  # not work on .py file due to ModuleNotFoundError
        subprocess.run(["python", f"{notebook}"])  # wait, do nothing until it finish