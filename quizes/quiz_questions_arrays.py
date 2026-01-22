"""
Questions for the Quick NumPy quiz
This module exports a single symbol `questions` which is a list
in the jupyterquiz `multiple_choice` / `answers` format.
"""

questions = [
    {
        "question": "What is the shape of np.array([1, 2, 3])?",
        "type": "multiple_choice",
        "answers": [
            {"code": "(3,)", "correct": True},
            {"code": "(1, 3)", "correct": False},
            {"code": "(3, 1)", "correct": False},
            {"code": "3", "correct": False},
        ],
    },
    {
        "question": "Given a = np.array([[1,2], [3,4]]), what is a.shape?",
        "type": "multiple_choice",
        "answers": [
            {"code": "(2, 2)", "correct": True},
            {"code": "(4, 1)", "correct": False},
            {"code": "(2,)", "correct": False},
            {"code": "(1, 2)", "correct": False},
        ],
    },
    {
        "question": "What does np.arange(10, 60, 5) produce?",
        "type": "multiple_choice",
        "answers": [
            {"code": "[10, 15, 20, 25, 30, 35, 40, 45, 50, 55]", "correct": True},
            {"code": "[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]", "correct": False},
            {"code": "[10, 60]", "correct": False},
            {"code": "[5, 10, 15, ..., 60]", "correct": False},
        ],
    },
    {
        "question": "If a = np.array([[1,2],[3,4]]), what does a[1, :] return?",
        "type": "multiple_choice",
        "answers": [
            {"code": "A rank-1 array with the second row", "correct": True},
            {"code": "A rank-2 array with shape (1, 2)", "correct": False},
            {"code": "The second column", "correct": False},
            {"code": "An error", "correct": False},
        ],
    },
    {
        "question": "Given a = np.array([1,8,9,-3,2,4,7,9]), what does a[a > 5] return?",
        "type": "multiple_choice",
        "answers": [
            {"code": "All elements greater than 5", "correct": True},
            {"code": "Indices of elements > 5", "correct": False},
            {"code": "A boolean mask", "correct": False},
            {"code": "An error", "correct": False},
        ],
    },
    # Additional questions (varied correct answer positions)
    {
        "question": "Which function creates an array of zeros with shape (2,3)?",
        "type": "multiple_choice",
        "answers": [
            {"code": "np.empty((2,3))", "correct": False},
            {"code": "np.zeros((2,3))", "correct": True},
            {"code": "np.ones((2,3))", "correct": False},
            {"code": "np.full((2,3), 0)", "correct": False},
        ],
    },
    {
        "question": "What is the dtype of np.array([1, 2.0, 3])?",
        "type": "multiple_choice",
        "answers": [
            {"code": "int64", "correct": False},
            {"code": "object", "correct": False},
            {"code": "float64", "correct": True},
            {"code": "bool", "correct": False},
        ],
    },
    {
        "question": "Which call returns 5 evenly spaced values between 0 and 1 inclusive?",
        "type": "multiple_choice",
        "answers": [
            {"code": "np.arange(0,1,0.2)", "correct": False},
            {"code": "np.linspace(0,1,5)", "correct": True},
            {"code": "np.linspace(0,1,4)", "correct": False},
            {"code": "np.arange(0,1,0.25)", "correct": False},
        ],
    },
    {
        "question": "What does a.shape return for a = np.array([[1,2,3],[4,5,6]])?",
        "type": "multiple_choice",
        "answers": [
            {"code": "(3, 2)", "correct": False},
            {"code": "6", "correct": False},
            {"code": "(2, 3)", "correct": True},
            {"code": "(2,)", "correct": False},
        ],
    },
    {
        "question": "Which operation performs elementwise multiplication between arrays x and y?",
        "type": "multiple_choice",
        "answers": [
            {"code": "x @ y", "correct": False},
            {"code": "np.dot(x,y)", "correct": False},
            {"code": "x * y", "correct": True},
            {"code": "x.dot(y)", "correct": False},
        ],
    },
    {
        "question": "Which function returns the indices that would sort an array?",
        "type": "multiple_choice",
        "answers": [
            {"code": "np.sort(a)", "correct": False},
            {"code": "np.argsort(a)", "correct": True},
            {"code": "np.argmax(a)", "correct": False},
            {"code": "a.sort()", "correct": False},
        ],
    },
]
