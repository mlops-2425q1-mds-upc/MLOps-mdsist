"""
Script to adapt pynblint to the precommit demands
"""

import json
import os
import sys
from typing import List

import typer

app = typer.Typer()


@app.command()
def main(
    train_set_path: List[str],
) -> None:
    """
    For each notebook file staged, execute pynblint and output the recomendations.
    The execution fails if at least one of the file analysises issues recomendations
    """

    problem = False

    for nb_file in train_set_path:
        code = os.system(f"pynblint -q {nb_file} -o test.json")
        if code == 0:
            with open("test.json", encoding="UTF-8") as file:
                jdoc = json.loads(file.read())
            os.remove("test.json")
            typer.echo("---------------------------------------------")
            typer.echo(jdoc["notebook_metadata"]["notebook_name"])
            typer.echo("---------------------------------------------\n")
            if len(jdoc["lints"]) > 0:
                for slug in jdoc["lints"]:
                    typer.echo(slug["slug"])
                    typer.echo("\t" + slug["description"])
                    typer.echo("\t" + slug["recommendation"] + "\n")
                problem = True

        else:
            typer.echo("Error")
            sys.exit(1)

    if problem:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    app()
