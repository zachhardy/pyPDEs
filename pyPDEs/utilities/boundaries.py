__all__ = ["Boundary", "DirichletBoundary", "NeumannBoundary",
           "RobinBoundary", "ReflectiveBoundary", "MarshakBoundary",
           "VacuumBoundary", "ZeroFluxBoundary"]

class Boundary:
    def __init__(self) -> None:
        self.type: str = None


class DirichletBoundary(Boundary):
    def __init__(self, bndry_val: float) -> None:
        super().__init__()
        self.type = "DIRICHLET"
        self.value: float = bndry_val


class NeumannBoundary(Boundary):
    def __init__(self, bndry_val: float) -> None:
        super().__init__()
        self.type = "NEUMANN"
        self.value: float = bndry_val


class RobinBoundary(Boundary):
    def __init__(self, a: float, b: float, f: float) -> None:
        super().__init__()
        self.type = "ROBIN"
        self.a: float = a
        self.b: float = b
        self.f: float = f


class ReflectiveBoundary(NeumannBoundary):
    def __init__(self) -> None:
        super().__init__(0.0)


class MarshakBoundary(RobinBoundary):
    def __init__(self, f: float) -> None:
        super().__init__(0.25, 0.5, 2.0 * f)


class VacuumBoundary(RobinBoundary):
    def __init__(self) -> None:
        super().__init__(0.25, 0.5, 0.0)


class ZeroFluxBoundary(DirichletBoundary):
    def __init__(self) -> None:
        super().__init__(0.0)
