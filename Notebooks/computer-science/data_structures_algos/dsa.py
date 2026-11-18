import marimo

__generated_with = "0.25.2"
app = marimo.App()

with app.setup:
    # Import libraries and set up the notebook in the initialisation

    # Imports for notebook set-up, animations and themes:
    import marimo as mo
    import manim as mn
    from mathsandmeander.common import bib, manim_tools, theme
    from mathsandmeander.common.widgets import PlaySlider
    from manim_slides.slide.manim import Slide
    import matplotlib.pyplot as plt
    import matplotlib.animation as anim
    theme.apply(usetex = True)

    # Imports for algorithm and data structures:
    from time import time
    from collections import deque
    from collections import defaultdict
    from collections import Counter


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Data Structures And Algorithms (in Python):

    By Donnell Obovu:

    ### I. Preface:

    This is an interactive notebook where I introduce data structures and algorithms in Python.

    The main reference for this notebook is [[Goodrich, Tamassia, Goldwasser 2013](#ref-DSAPython)].

    ### II. Prerequisites:

    For this notebook, I believe no prerequisites will be required, but a knowledge of:
    - Programming basics (variable assignment and operations)
    - Mathematical Induction
    - Basic Real Analysis
    would be help with understanding certain concepts quicker.

    ### III. Contents:
    1. Python Foundations:
       - 1.1. Built-in Types, Mutability and References
       - 1.2. Iterators, Generators and Comprehensions
       - 1.3. Classes and Dunder Methods (container, comparison, hashing protocols)
    2. Algorithm Analysis:
       - 2.1. Experimental vs Theoretical Analysis
       - 2.2. Asymptotic Notation ($O$, $\Omega$, $\Theta$)
       - 2.3. Worst, Average and Amortised Cost
       - 2.4. Recurrences and Correctness (induction, loop invariants)
    3. Core Algorithms:
       - 3.1. Recursion
       - 3.2. Searching (linear, binary, bisect)
       - 3.3. Sorting (insertion, merge, quick; lower bound; sorted/key)
       - 3.4. Selection (quickselect)
    4. Data Structures:
       - 4.1. Arrays and Strings
       - 4.2. Linked Lists
       - 4.3. Stacks, Queues and Deques
       - 4.4. Hash Tables, Sets and Maps
       - 4.5. Trees (binary, traversals, BST; trie)
       - 4.6. Heaps and Priority Queues
       - 4.7. Graphs (representations, BFS, DFS, topological sort, Dijkstra, union-find)
    5. Problem-Solving Techniques:
       - 5.1. Two Pointers
       - 5.2. Sliding Window
       - 5.3. Prefix Sums
       - 5.4. Backtracking
       - 5.5. Greedy
       - 5.6. Dynamic Programming
       - 5.7. Choosing a Technique
    6. Appendix: Python Toolkit Cheat Sheet
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Python Foundations:

    ### 1.1. Built-in Types:
    Python comes equipped with some built-in classes. Below is a table of the key built-in classes, a description of what they are, their mutability and their default type.
    | Class | Description | Mutability | Default |
    | ----- | ----------- | -----------| ------- |
    | **bool** | Boolean Value | **N** | **False** |
    | **int** | Integer (arbitrary magnitude) | **N** | 0 |
    | **float** | Floating-point number | **N** | 0.0 |
    | **list** | Mutable sequence of objects | **Y** | `[]` (empty list) |
    | **tuple** | Immutable sequence of objects | **N** | `()` (empty tuple) |
    | **str** | Character string | **N** | `''` (empty string) |
    | **set** | Mutable unordered collection of distinct objects | **Y** | `set()` (empty set) |
    | **frozenset** | Immutable unordered collection of distinct objects | **N** | (empty (frozen) set) |
    | **dict** | Associative (hash) map | **N** | `{}` (empty dictionary) |
    | **frozendict** | An immutable associative (hash) map (only available in latest versions of Python) | **Y** | (empty frozen dictionary) |

    The **bool** class:
    - used to manipulate logical (boolean) values.
    - only two values are ` True` and `False`
    - default constructor, `bool()`, returns `False`.
    - `bool(x)` evaluates to `False` if:
      - `x` is a container type (e.g. strings and lists) and empty
      - `x` is a numerical type (e.g. floats and ints) and is 0.
    - `bool(x)` evaluates to `True` if:
      - `x` is a non-empty container type
      - `x` is a non-zero number

    The **int** class:
    - Stores integers
    - Default constructor, `int()` returns 0.
    - `int(f)` for a float `f` returns a truncated value of f
    - `int(s)` can parse a string presumed to represent an integer.
    - We can consider integers of different bases by using the refix `0` followed by the base, and the representation of the number in that base. e.g. `0b1011` is the binary representation of 21. `0x` and `0o` prefixes are used for hexadecimal and octal respectively.

    The **float** class:
    - `float()` returns 0.
    - `float(i)` returns the float representation of an integer i. E.g. returns `2.0` if `i == 2`

    Sequence types:
    - The **list**, **tuple** and **str** classes are sequence types.
    - Sequence types are a store of values in which the order is significant.

    The **list** class:
    - A _referential_ strcture, it stores a sequence of references to its elements.
    - _array based_ and _mutable_
    - Can dynamically expand and contract as needed.
    - Delimited with `[]`
    - `list()` returns the empty list, `[]`.
    - The constructor `list()` will accept any iterable type.

    The **tuple** class:
    - Immutable version of a sequence.
    - Delimited with `()`
    - To express a tuple of length one, we place a comma after the object in the tuple. E.g. `(a,)`

    The **str** class:
    - An immutable sequence of characters
    - Delimited by single quotes `''` or double quotes `"\"`
    - We have _raw_ strings, denoted with a `r' '` or `r" "` for character sequences containing certain special characters which otherwise serve a function.
    - We have _f_-strings, denoted `f'{var}'` to write more legible code when we want a string that depends on some variable.

    The **set** and **frozenset** classes:
    - Python's set class represents the mathematical idea of a set: a collection of elements, without duplicates and without an inherent order to those elements.
    - Main advantage of sets and frozensets:
      - Highly optimised method for checking whether a specific element is in the set.
    - Based on a data structure known as a ***hash table***.
    - Only instances of *immutable* types can be added to a set:
      - Sets and frozensets can store: ints, floats, strings, tuples, booleans and frozensets.
      - Sets and frozensets cannot store: lists and sets.
    - Sets are delimited with `{}`.
    - `{}` does _not_ denote the empty set. Use `set()` for the empty set.
    - `set(iterable)` returns a set containing the distinct elements of the iterable.

    The **dict** class:
    - represents a dictionary or mapping from a distinct set of keys to associated values.
    - Delimited by `{}`. `{}` returns an empty dictionary.
    - Nonempty dictionaries are expressed using comma-separated key:value pairs.
    - The keys must be unique, the values needn't be.
    - The constructor for dict is `dict()` and can take an existing mapping as a parameter, or it can take key:value pairs as a list containing elements of the form `(key, value)`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1.2. Iterators, Generators and Comprehensions:


    ### 1.3. Object-Oriented Programming:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Algorithm Analysis:
    An ***Algorithm*** is a step-by-step procedure for performing some task in a finite amount of time [[Goodrich, Tamassia, Goldwasser 2013](#ref-DSAPython)].

    It is possible for us to have two or more algorithms that perform the same task, which begs the question: _which algorithm is best for our use case?_

    We therefore need a way to evaluate algorithms and compare them. We do this by finding the amount of resources, usually _time_ or _space/memory_ needed for the algorithm to execute the algorithm for an input of a given size.

    Typically, the fewer resources needed for an algorithm to execute, the better, or more _efficient_ we say an algorithm is.

    ### 2.1. Experimental vs Theoretical Analysis:

    #### 2.1.1. Experimental Analysis:
    We can study an algorithm's runtime by executing it on various test inputs and recording the time spent during each execution.

    Python allows us to do this easily, using the time library:
    ```python
    from time import time

    start_time = time()
    algorithm(test_input)
    end_time = time()
    time_elapsed = end_time - start_time

    ```

    This essentially starts a stopwatch as the algorithm begins running on an input, and stops it once it finishes running.

    There are limitations to performing our analysis this way:
    - *Environment:* two algorithms are difficult to compare directly unless they are both run on the same hardware with the same software.
    - *Number of inputs:* we can only conduct experiments on a limited number of inputs.
    - *Implementation:* an algorithm has to be fully implemented in order to execute it and study its running time experimentally.

    #### 2.1.2. Theoretical Analysis:

    We require non-experimental methods for analysing the efficiency of algorithms. Our approach should:
    - Allow us to evaluate an algorithm's efficiency independent of hardware or software environment
    - Take into account all possible inputs
    - Be performed by studying a high-level description of the algorithm without need for implementation.

    The theoretical approach for analysing an algorithm's runtime involves taking a high-level description of the algorithm and counting the number of _primitive operations_ taken for the algorithm to complete its execution given a specific input.

    A ***primitive operation*** corresponds to a low-level instruction which has constant execution time. Examples include:
    - Assigning an indentifier to an object
    - Determining the object associated with an identifier
    - Performing an arithmetic operation
    - Comparing two numbers
    - Accessing a single element of a sequence or container
    - Calling a function (excluding operations executed within the function)
    - Returning from a function

    For analysing an algorithm's space efficiency, we can count how much data is required to be stored for the algorithm to run on a given input [UNCORROBORATED].

    To capture the runtime or memory demand of an algorithm, we associate with each algorithm a function $f(n)$, that characterises the number of primitive operations or spaces in memory that are required to be performed, for an input of a given size $n$. The issue is, an algorithm may have different runtimes or memory demands for different inputs even if they re of the same size. Thus we perform our analysis on the _worst-case scenario_.

    ### 2.2. Asymptotic Notation (Big O, Big Omega and Big Theta):

    We use _Big-O, Big-Omega_ and _Big-Theta_ notation to characterise the behaviour of an algorithm as input size $n\to\infty$.

    ***Big O:*** Let $f,g:\mathbb{N}\to\mathbb{R}$, we say that $f(n)$ is $O(g(n))$ if there is a real constant $c>0$ and an integer constant $n_0$ such that $$f(n)\leq cg(n),\text{ for }n\geq n_0.$$

    ***Big Omega:*** we say that $f(n)$ is $\Omega(g(n))$ if there is a real constant $c>0$ and an integer constant $n_0$ such that $$f(n)\geq cg(n),\text{ for }n\geq n_0.$$

    ***Big Theta:*** we say that $f(n)$ is $\Theta(g(n))$ if there are real constants $c_{1},c_{2}>0$ and an integer constant $n_0$ such that $$c_{1}g(n)\leq f(n)\leq c_{2}g(n),\text{ for }n\geq n_0.$$


    We will proceed with Big O notation from here on for our analysis unless otherwise stated.

    We use Big O notation to characterise a function's asymptotic behaviour as closely as possible. E.g. $f(n) = 3n^2 + n + 1$ is $O(n^6)$ and $O(n^3)$ but its best characterise $f$ as $O(n^2)$.

    | Functions $f(n)$ | Examples |
    |------------------| ----- |
    | $O(1)$ | Getting the first element of a list, as we jump straight to the memory location. It is always on step regardless of the size of the list. |
    | $O(\log n)$  | Binary search on a sorted list, as each iteration halves the range. |
    | $O(n)$       | Finding the maximum in an unsorted list, as we must examine each element once. Doubling the input doubles the work |
    | $O(n\log n)$ | "Divide and Conquer" algorithms such as Merge sort. Splitting the input into half (the $\log$ part), and doing a linear amount of work at each level of the splitting (the $n$ part) |
    | $O(n^{k})$   | $k$ levels of nested loops. In a double nested loop ($k=2$), the highest order loop passes through each element, and on each element, the lower order loop passes through each of the $n$ elements. $n\times n = n^2$.|
    | $O(b^n)$     | Brute-force guessing every password of length $n$ from an alphabet of $b$ characters. There are $b^n$ possible guesses.|

    The order of size is $$O(1)<O(\log n)<O(n)<O(n^{k})<O(b^n).$$

    In general, asymptotically smaller runtimes are better, however Big O (and similar) notation hide constant and lower order terms that feature in an algorithm's runtime. This is something to be careful of. For example, an $O(n\log n)$ algorithm, A, is asymptotically worse than an $O(n)$ algorithm, B, but if Algorithm B has runtime $T_{B}(n) = 10^{100}n$ and Algorithm A has runtime $T_{A}(n) = n\log n$, then for all practical purposes, algorithm A is preferable to $B$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Core Algorithms:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Data Structures:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Problem-Solving Techniques:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Appendix: Python Toolkit Cheat Sheet:
    """)
    return


@app.cell
def _():
    bib.bibliography(__file__)
    return


if __name__ == "__main__":
    app.run()
