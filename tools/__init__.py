"""
IPython Tools taken from Jean-Luc Stevens TCAL repository.
"""

import os, sys, time, difflib

from IPython.core import page
from holoviews import NdMapping
from holoviews.interface.collector import AttrTree

from io import BytesIO
from contextlib import contextmanager

#=============#
# Build index #
#=============#

def find_notebooks(root, exclude_dirs):
    """
    Find the IPython notebooks available from root, excluding specific
    notebooks and directories. Used by build_index function.
    """
    notebooks = []
    for basedir, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f.endswith(".ipynb"):
                notebooks.append(os.path.join(basedir, f))
    return notebooks

def build_index(name, root, index, exclude=[], exclude_dirs=[]):
    """
    Given a root directory and an index (a dictionary of notebook
    names and descriptions) build an index of the available notebooks,
    excluding any mentioned in the exclude list.
    """
    def mtime(f): return "Modified: %s" % time.ctime(os.path.getmtime(f))
    def ctime(f): return "Created: %s" % time.ctime(os.path.getctime(f))

    def timestamps(f):
        return "</br></br><p style='font-size:small;'>%s</br>%s</p></br>" % (ctime(f),mtime(f))

    def anchor(name):
        return "<a href='#%s'>^</a>" % name

    root = os.path.abspath(root)
    exclude_dirs = ['.git', '.ipynb_checkpoints'] + exclude_dirs
    exclude = exclude + ['Index'] + ['Untitled%d' % i for i in range(10)]
    notebooks = find_notebooks(root, exclude_dirs)
    info = [("%s <a href='%s'>%s</a>"%  (anchor(os.path.basename(el)[:-6]),
                                         './' + os.path.relpath(el, root),
                                         os.path.basename(el)[:-6]),
             ('<b><i>' + index.get(os.path.basename(el)[:-6], '--') + '</b></i>'
              + timestamps(el))) for el in notebooks if os.path.basename(el)[:-6] not in exclude]

    contents = ' '.join("<dt>%s</dt><dd>%s</dd>" % (name, desc) for (name,desc) in info)
    html = "<p style='font-size:large;'></br><b>Notebooks in %r</b></p>" % name
    html +="<dl class='dl-horizontal' style='font-size:large;'>%s</dl>" % contents
    return html


#===================#
# Caching decorator #
#===================#
import hashlib, pickle


def cache(val, cache_dir='./cache'):
    """
    Save a value to the cache (i.e as a pickle).

    Note that you can use this function directly to save a value and
    decorate a function with the cached decorator after discovering
    some loop is particularly slow to execute.
    """
    cache_dir = os.path.abspath(cache_dir)
    pickle_str = pickle.dumps(val)
    md5obj = hashlib.md5()
    md5obj.update(pickle_str)
    hashval = md5obj.hexdigest()
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, hashval), 'w') as f:
        f.write(pickle_str)
    print("md5=%r" % hashval)


def cached(md5=None, cache_dir='./cache', verbose=False):
    """
    Decorator that memoizes the result in a pickle stored in the
    cache_dir directory. If md5 is None, the data is regenerated and
    the suitable md5 hash is generated.
    """
    cache_dir = os.path.abspath(cache_dir)
    md5_path = os.path.join(cache_dir, md5) if md5 else None

    def wrapper(fn):
        def wrapped(*args, **kwargs):
            if md5_path is None:
                val = fn(*args, **kwargs)
                cache(val, cache_dir)
                return val
            elif os.path.isfile(md5_path):
                if verbose: print("[Loading data from cached pickle]")
                return pickle.load(open(md5_path, 'r'))
            else:
                print("Could not find file with md5 hash %r in cache" % md5)
                print("Set md5 argument to None to regenerate pickle.")
        return wrapped
    return wrapper



#===================#
# Diffing functions #
#===================#

@contextmanager
def stdout_redirected(new_stdout):
    "Example taken from PEP 0343"
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout


def show_diff(file1, file2, diff_cmd='git diff -w --word-diff=color'):
    """
    Given two tracked files show the coloured, word-level diff in the
    IPython pager using ``git diff`` for specific commits.

    The specified files should include a commit reference as well,
    e.g. <SHA>:./path/model.ty HEAD:reference.ty to allow diffs
    against historical file versions. If file2 has a reference, file1
    needs one as well (HEAD may be used.)

    Also note that you can use --no-index to diff untracked files but
    then the use of commit references will be disabled.

    Default command requires git 1.8 and above.
    """
    if not os.path.isfile(file1.split(':')[-1]):
        raise IOError("File %r not found." % file1)
    if not os.path.isfile(file2.split(':')[-1]):
        raise IOError("File %r not found." % file2)

    if '--no-index' not in diff_cmd and ':' not in file1+file2:
        raise Exception('Reference required unless --no-index used.')

    diff = get_ipython().getoutput(' '.join([diff_cmd, file1, file2]))
    if ''.join(diff).startswith('error'):  print '\n'.join(diff)
    else: page.page('\n'.join(diff[1:]))



def missing_runs(args1, args2, exclude=[]):
    """
    Diffing utility to find the difference between the specs of two
    Argument specifiers.

    Particularly useful for finding out what jobs are missing after
    execution on Eddie. In this case, the first args should be a
    FileInfo with the target files and the second should be an Args
    object representing the intended parameter space.

    The exclude argument excludes keys from show(), e.g to omit the
    filename key in a FileInfo Args object.
    """
    with BytesIO() as b1:
        with stdout_redirected(b1):
            args1.show(exclude=exclude)
        b1.seek(0)
        s1 = b1.read()

    with BytesIO() as b2:
        with stdout_redirected(b2):
            args2.show(exclude=exclude)
        b2.seek(0)
        s2 = b2.read()

    lines = []
    # Trim out the spec line numbers
    s1_lines = [' '.join(l.split()[1:]) for l in s1.splitlines(1)]
    s2_lines = [' '.join(l.split()[1:]) for l in s2.splitlines(1)]
    lines = [l for l in difflib.unified_diff(s1_lines, s2_lines,
                                    fromfile='First:\n\n'+str(args1),
                                    tofile=  'Second:\n\n'+str(args2))]
    diff = False if len(lines) == 0 else True
    heading = "Difference in specs"
    lines = [heading, '='*len(heading), ""] + lines
    if not diff: lines.append("No difference detected")
    page.page('\n'.join(lines))


def show_history(filename,
                 cmd=['git', 'log', '--pretty=format:"%h    %ad    %s"', '-n', '15']):
    """
    Display the log history of a specific file in a compact format.
    """
    diff = get_ipython().getoutput(' '.join(cmd+[filename]))
    print("\n".join(diff))


#=============================#
# Collector library functions #
#=============================#

def save_collector(collector, name, description, readme_file):
    """
    Save a collector as a pickle with the given name and append
    information about the collector to the specified README file
    (pickle assumed to be in the same folder).

    Stores the description and repr in the pickle separate from the
    collector data itself.
    """
    readme_file = os.path.abspath(readme_file)
    (path, _) =  os.path.split(readme_file)

    pkl_path = os.path.join(path, '%s.pkl' % name.replace(' ', '_'))
    if os.path.isfile(pkl_path):
        raise IOError("%r already exists." % os.path.basename(pkl_path))

    with open(pkl_path, 'w') as f:
        pickle.dump(description, f)
        pickle.dump(repr(collector), f)
        pickle.dump(collector, f)

    with open(readme_file, 'a') as readme:
        empty_line = ''
        heading = "Collector %r" % name
        underline = '-' * len(heading)

        lines = [heading, underline, empty_line, description, empty_line,
                 str(collector), empty_line, empty_line]

        readme.write('\n'.join(lines))


def load_collector(name, directory, include_repr=False):
    """
    Load a collector by name from the given directory, printing the
    pickle description. If the pickle fails to load, the pickle repr
    is still available.
    """
    name = name.replace('.pkl', '')
    pkl_path = os.path.join(directory, '%s.pkl' % name.replace(' ', '_'))
    if not os.path.isfile(pkl_path):
        raise IOError("File %r not found." % pkl_path)

    with open(pkl_path, 'r') as f:
        description = pickle.load(f)
        collector_repr = pickle.load(f)
        try:
            collector = pickle.load(f)
            print("Loaded collector %r : %s" % (name, description))
        except:
            collector = collector_repr
            print("Could not unpickle collector %r. Returning the repr instead.")

    return collector


#=========================#
# Data handling utilities #
#=========================#

def select(filedata, cache={}, verbose=False, **kwargs):
    """
    Convenience function for selecting and loading AttrTrees from
    files generated by Lancet (e.g. on Eddie). Given a FileInfo object
    set to load .view files, select a subset of the files via kwargs,
    load the specified data and return the results as a suitable
    AttrTree. Note that the selection arguments are stored under the
    'Metadata' path of the returned AttrTree.

    This function is designed to be as cheap to use as possible: files
    are selected *before* data is loaded and this function implements
    a cache. The first time a particular selection is used, the data
    will be loaded and collated (slow) but subsequent uses of the same
    selection are fast (cache accesses).

    The cache argument can be set to None to disable the cache (or
    cache={} can be used once to clear the cache). Setting
    verbose=True prints messages when the cache is accessed.

    Note that the cache uses the id of the input FileInfo object which
    means the cache will not be used across different FileInfo
    arguments.
    """
    if isinstance(cache, dict):
        cached = cache.get(id(filedata), {})
        cache_key = tuple(sorted(kwargs.items()))
        if cache_key in cached:
            if verbose:  print "Returning cached value."
            return cached[cache_key]

    selected = filedata.ndmapping.select(**kwargs)
    if not isinstance(selected, NdMapping):
        key = tuple([kwargs[l] for l in filedata.ndmapping.dimension_labels])
        selected = filedata.ndmapping.clone(items=[(key, selected)])

    tree = AttrTree.merge(filedata.load(selected, data_key='data').values())
    tree.fixed = False
    tree.set_path('Metadata', kwargs)

    if isinstance(cache, dict):
        if id(filedata) not in cache:
            cache[id(filedata)] = {}
        cache_key = tuple(sorted(kwargs.items()))
        cache[id(filedata)][cache_key] = tree

    return tree
