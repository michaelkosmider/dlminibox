from dl import Parameter


class Module:
    def __init__(self):
        self._child_modules = {}
        self._params = {}
        self._hyper_parameters = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._params[name] = value

        elif isinstance(value, Module):
            self._child_modules[name] = value

        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        rep = f"{self.__class__.__name__}"
        if len(self._hyper_parameters) > 0:
            rep = (
                rep
                + "("
                + ", ".join(
                    [
                        f"{param} = {repr(value)}"
                        for param, value in self._hyper_parameters.items()
                    ]
                )
                + ")"
            )

        return rep

    def parameters(self):
        parameter_flatlist = list(self._params.values())

        for child_module in self._child_modules.values():
            parameter_flatlist.extend(child_module.parameters())

        return parameter_flatlist

    # Resolve: it is unclear what the printing order is, other than inner modules appearing less indented. Figure this out.
    # Resolve2: Apparently dicts are ordered
    def print(self, indent=0):
        if len(self._child_modules) == 0:
            print(repr(self))

        if len(self._child_modules) > 0:
            print(self.__class__.__name__)
            for name, child_module in self._child_modules.items():
                print(f"{'    ' * (indent + 1)}{name} : ", end="")
                child_module.print(indent + 1)
