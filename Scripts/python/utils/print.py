import sys

def print2(element, id=None, print_type=True, sep=' ', end='\n', file=sys.stdout, flush=False, activate=True):
    if activate:
        if id:
            print("{} :\n{}".format(id ,element), sep=sep, end=end, file=file, flush=flush)
        else:
            print(element, sep=sep, end=end, file=file, flush=flush)
        
        if print_type:
            print("type: {}".format(type(element)))

        print("")
