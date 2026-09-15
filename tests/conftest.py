import pathlib
import sys

# Permite `import code...` al ejecutar pytest desde la raiz del repo.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
