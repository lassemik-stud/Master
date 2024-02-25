from module.restructure_db.create_problem_pan13 import test_create_pan13_problem

for i in range(1, 100):
    test_create_pan13_problem(10, 3, i)