from node import TreeNode

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def recursive_sum(lst):
    if not lst:
        return 0
    return lst[0] + sum(lst[1:])

def find_maximum(root):
    if not root:
        return -1
    else:
        return max((root.value, find_maximum(root.left), find_maximum(root.right)))

def tree_to_list(root):
    if not root:
        return []
    else:
        return [root.value] + tree_to_list(root.left) + tree_to_list(root.right)

def equals(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False
    if root1.value != root2.value:
        return False
    return equals(root1.left, root2.left) and equals(root1.right, root2.right)


if __name__ == '__main__':
    # build a tree
    #     1
    #    / \
    #   2   3
    #  /
    # 4
    t1 = TreeNode(1)
    t1.left = TreeNode(2)
    t1.right = TreeNode(3)
    t1.left.left = TreeNode(4)

    # build a tree
    #     1
    #    / \
    #   2   3
    #  /   /
    # 4   5
    t2 = TreeNode(1)
    t2.left = TreeNode(2)
    t2.right = TreeNode(3)
    t2.left.left = TreeNode(4)
    t2.right.left = TreeNode(5)

    print find_maximum(t1)    # 4
    print find_maximum(t2)    # 5
    print equals(t1, t2)      # False
