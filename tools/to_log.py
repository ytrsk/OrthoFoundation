log_file = None
local_rank = 0


def set_file(file, rank):
    global log_file, local_rank
    log_file = file
    local_rank = rank


def to_log(s: str, print_once=True):
    if (not print_once) or local_rank == 0:
        print(s)
        if log_file is not None:
            print(s, file=log_file)
            log_file.flush()