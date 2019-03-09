import pickle

def save_pickle(filename, obj):
    """ Save a python object to disk
    pickle files usually have a .p or .pickle extension
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(pickle_file):
    """ Load a pickled (saved) python object
    """
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)
