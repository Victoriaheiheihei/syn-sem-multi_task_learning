def get_task_batch(step, task_type):
    if task_type == 'task':
        while True:
            step += 1
            yield str(step) + task_type
    else:
        while True:
            step += 1
            yield str(step) + task_type

def test_gtb():
    a = get_task_batch(1, 'task')
    print(next(a))
    print(next(a))
    print(next(a))

def test_fct(t=1):
    def t1():
        print(t)
    t1()

def test_dq():
    from _collections import deque
    checkpoint_queue = deque([], maxlen=50)
    checkpoint_queue.append('1')
    checkpoint_queue.append('2')
    checkpoint_queue.append('3')
    print(checkpoint_queue)
    if '1' in checkpoint_queue:
        print('yes')
    else:
        print('nope')
    if '4' in checkpoint_queue:
        print('yes')
    else:
        print('nope')

if __name__ == "__main__":
    test_dq()