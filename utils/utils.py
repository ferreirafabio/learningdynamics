import argparse
import os
import re
import collections

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def chunks(l, n):
  """Yield successive n-sized chunks from l.
  Used to create n sublists from a list l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def get_file_paths_from_dir(source_path, file_type=".npz"):
    file_paths = {}
    for root, dirs, files in os.walk(source_path):

        trial_idx = os.path.basename(root)
        if trial_idx != "source":
            dct = {}
            file_paths[trial_idx] = dct

        for file in files:
            if file.endswith(file_type):
                n = re.search("\d+", file).group(0)
                if n in dct.keys():
                    lst = dct[n]
                    lst.append(os.path.join(root, file))
                    dct[n] = lst
                else:
                    lst = []
                    lst.append(os.path.join(root, file))
                    dct[n] = lst

    # sort inner and outer dicts to avoid conflicts in tfrecord generation
    for i in file_paths.keys():
        file_paths[i] = collections.OrderedDict(sorted(file_paths[i].items(), key=lambda x: chr(int(x[0]))))

    file_paths = collections.OrderedDict(sorted(file_paths.items(), key=lambda x: chr(int(x[0]))))

    return [list(file_paths[i].values()) for i in file_paths.keys()]



def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval