class Pipe:
    def __init__(self):
        self.value = None
        self.functions = []

    def __or__(self, other):
        if callable(other):
            self.functions.append(other)
        else:
            raise TypeError("Right operand must be a callable.")
        return self

    def invoke(self, value):
        self.value = value
        for func in self.functions:
            self.value = func(self.value)
        return self.value


def func1(value):
    return value + 1


def func2(value):
    return value + 2


def func3(value):
    return value + 3


obj = Pipe()
# Create the function chain
chain = (obj | func1 | func2 | func3)

# Set the initial value and invoke the chain
result = chain.invoke(1)
print(result)  # Output should be 7 (1 + 1 + 2 + 3)
