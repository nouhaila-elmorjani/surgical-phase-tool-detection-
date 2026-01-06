import io
import os
import tokenize


def iter_python_files(base_dir: str):

    for dirpath, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", "__pycache__", ".venv", "venv", "env"}
        ]
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def remove_comments_from_file(path: str):

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    filtered_tokens = [tok for tok in tokens if tok.type != tokenize.COMMENT]
    new_source = tokenize.untokenize(filtered_tokens)

    if new_source != source:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_source)


def main():

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for path in iter_python_files(base_dir):
        if os.path.basename(path) == os.path.basename(__file__):
            continue
        remove_comments_from_file(path)


if __name__ == "__main__":
    main()
