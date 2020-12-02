if __name__ == "__main__":

    import pathlib
    import shutil
    import subprocess
    import sys
    import os


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

        # Convert what just ran to html in .reports.
        subprocess.run([
            'jupyter',
            'nbconvert',
            '--to', 'html',
            '--output-dir=.reports',
            notebook.name,
        ])

        # delete notebooks after run completion
        os.remove(notebook.name)

    elif notebook.name.split('.')[-1] == 'py':
        subprocess.run(["python", "./src/proc/200 train.py"])  # wait, do nothing until it finish