import errno
import os
import os.path

from tabulate import tabulate


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def open_path(path, mode):
    mkdir(os.path.dirname(path))
    return open(path, mode, encoding="utf-8")


def read_file_to_list(path):
    file = open_path(path, "r")
    lines = file.readlines()
    strings = []

    for line in lines:
        line = line.encode('ascii', 'ignore').decode("utf-8")
        strings.append(line.rstrip())

    return strings


def read_file(path):
    file = open_path(path, "r")
    text = file.readlines()

    return text[0].encode('ascii', 'ignore').decode("utf-8")


def read_sentences_from_file(path):
    file = open_path(path, "r")
    text = file.readlines()
    for index, sentence in enumerate(text):
        text[index] = sentence.replace('\n', '')

    return text


def write_list_to_file(path, list):
    file = open_path(path, "w+")
    file.write(tabulate(list, tablefmt='orgtbl'))
    file.close()


def write_text_to_file(path, text):
    file = open_path(path, "w+")
    file.write(text)
    file.close()
