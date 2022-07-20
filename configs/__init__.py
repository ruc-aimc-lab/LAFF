import os

if __name__ == '__main__':
    print((os.listdir('./')))
    for each in os.listdir('./'):
        if 'pyc' in each:
            continue
        if 'w2v' in each:
            each = each.replace('.py', '')
            print(each)