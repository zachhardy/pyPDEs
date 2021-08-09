from . import GaussLegendre


class LineQuadrature(GaussLegendre):
   def __init__(self, order: int = 2) -> None:
       super().__init__(order)
