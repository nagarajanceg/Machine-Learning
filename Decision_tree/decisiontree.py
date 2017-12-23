import pandas as pd
import math
from collections import deque
import copy
import random
import sys

df = pd.read_csv(sys.argv[1])
validation_set = pd.read_csv(sys.argv[2])
test_set = pd.read_csv(sys.argv[3])

class node:
    def __init__(self, name, left, right, label):
        self.name = name
        self.left = left
        self.right = right
        self.label = label
    def reflect_x(self):
        return node(self.name, self.left, self.right, self.label)

def infoGain(overallEntropy, expectedEntropy):
    gain = overallEntropy - expectedEntropy
    return gain

def calculateEntropy(entityZero, entityOne):
    return -(entityZero) * math.log(entityZero, 2) - (entityOne) * math.log(entityOne, 2);

## Entropy calculation
def calculate_entity(class_zero_count, class_one_count, total_count):
    if class_zero_count == 0 or class_one_count == 0 or total_count == 0 :
        return 0
    entityZero = float(class_zero_count) / float(total_count)
    entityOne = float(class_one_count) / float(total_count)
    return calculateEntropy(entityZero, entityOne)

def null_check(element):
    if element is None:
        return True
    else:
        return False

def display_tree(node, level):
    if null_check(node):
        return
    if type(node.name) is int:
        print(node.name, end="\n")
    if not null_check(node.left):
        print("\n", "|" * level, node.name, "=", 0, ":",end=" " )
        display_tree(node.left, level + 1)
    if not null_check(node.right):
        print("|" * level, node.name, "=", 1, ":",end=" ")
        display_tree(node.right, level + 1)
## count the class based on value
def countby_class(dataset, value):
    data = dataset["Class"]
    count = 0
    for x in data:
        if x == value:
            count += 1
    return count

def variance_impurity(dataset, total_count):
    class0 = countby_class(dataset, 0)
    class1 = countby_class(dataset, 1)
    if class0 == 0 or class1 ==0:
        return 0
    else:
        value = (float(class0)/float(total_count) * float(class1)/float(total_count))
        return value

def featureEntropy(dataset, col_name, overallCount):
    ##grouping by available class
    classZeroCount, classOneCount = dataset.groupby(["Class"]).size()
    overallEntropy = calculate_entity(classZeroCount, classOneCount, overallCount)

    # assign two d array to hold all possible values
    data = [[] for _ in range(2)]

    val_0 = dataset.loc[dataset[col_name] == 0]
    val_1 = dataset.loc[dataset[col_name] == 1]

    if countby_class(val_0, 0) != 0:
        data[0].append(countby_class(val_0, 0))
    else:
        data[0].append(0)
    if countby_class(val_0, 1) != 0:
        data[0].append(countby_class(val_0, 1))
    else:
        data[0].append(0)
    if countby_class(val_1, 0) != 0:
        data[1].append(countby_class(val_1, 0))
    else:
        data[1].append(0)
    if countby_class(val_1, 1) != 0:
        data[1].append(countby_class(val_1, 1))
    else:
        data[1].append(0)

    classZero = data[0][0] + data[0][1]
    classOne = data[1][0] + data[1][1]

    et0 = calculate_entity(data[0][0], data[0][1], classZero)
    et1 = calculate_entity(data[1][0], data[1][1], classOne)
    if overallCount != 0:
        exp0 = classZero/overallCount
        exp1 = classOne/overallCount
    else:
        exp0 = 0
        exp1 = 0
    info_gain = overallEntropy - (exp0 * et0) - (exp1 * et1)
    return info_gain

def intermediateGain(zero, one, total_count):
    entropy = calculate_entity(zero, one, total_count)
    return entropy

def gain_variance(dataset, cname):
    dataset0 = dataset.loc[dataset[cname] == 0]
    dataset1 = dataset.loc[dataset[cname] == 1]
    dataset0_length = len(dataset0)
    dataset1_length = len(dataset1)

    total_count = len(dataset)
    vi0 = variance_impurity(dataset0, dataset0_length)
    vi1 = variance_impurity(dataset1, dataset1_length)

    k0 = dataset0_length / total_count
    k1 = dataset1_length / total_count

    first_value = k0 * vi0
    second_value = k1 * vi1

    vi = variance_impurity(dataset, total_count)
    gain = vi - first_value - second_value
    return gain

##Determine the next attribute
def find_attr(dataset, type):
    col_names = list(dataset.columns.values)
    col_names.remove("Class")
    gain = []
    ## type true for information gain false- for variance impurity
    if type:
        overallCount = len(dataset['Class'])
        for cname in col_names:
            gain.append(featureEntropy(dataset, cname, overallCount))
    else:
        for cname in col_names:
            gain.append(gain_variance(dataset, cname))
    maxIndex = gain.index(max(gain))
    return col_names[maxIndex]

## recursively segerate data based on class values and maintained in a tree
def frameNewData(dataset):
    ##grab the present column names in the changed dataset
    cnames = list(dataset.columns.values)
    if len(cnames) == 1 :
        l1 = len(dataset.loc[dataset['Class'] == 1])
        l2 = len(dataset.loc[dataset['Class'] == 0])
        if l1 > l2:
            return node(1, None, None, 0)
        else:
            return node(0, None, None, 0)
    elif countby_class(dataset, 1) == 0:
        return node(0, None, None, 0)
    elif countby_class(dataset, 0) == 0:
        return node(1, None, None, 0)
    attr = find_attr(dataset, True)
    ##remove the selected attribute to avoid in further processing
    cnames.remove(attr)
    values_0 = dataset.loc[dataset[attr] == 0]
    values_1 = dataset.loc[dataset[attr] == 1]
    data_changed_set0 = values_0[cnames]
    data_changed_set1 = values_1[cnames]
    return node(attr, frameNewData(data_changed_set0), frameNewData(data_changed_set1),0)

## construct decision tree with variance
def frame_data_with_variance(dataset):
    cnames = list(dataset.columns.values)
    if len(cnames) == 1:
        l1 = len(dataset.loc[dataset['Class'] == 1])
        l2 = len(dataset.loc[dataset['Class'] == 0])
        if l1 > l2:
            return node(1, None, None, 0)
        else:
            return node(0, None, None, 0)
    elif countby_class(dataset, 1) == 0:
        return node(0, None, None, 0)
    elif countby_class(dataset, 0) == 0:
        return node(1, None, None, 0)
    ##select attribute as a nextnode
    attr_variance = find_attr(dataset, False)
    cnames.remove(attr_variance)
    values_0 = dataset.loc[dataset[attr_variance] == 0]
    values_1 = dataset.loc[dataset[attr_variance] == 1]
    data_changed_set0 = values_0[cnames]
    data_changed_set1 = values_1[cnames]
    return node(attr_variance, frame_data_with_variance(data_changed_set0), frame_data_with_variance(data_changed_set1),0)

def check_tree(element, tree_node):
    element_value = element[tree_node.name]
    if type(tree_node.name) is int:
        return tree_node.name
    if element_value == 0:
        return check_tree(element, tree_node.left)
    elif element_value == 1:
        return check_tree(element, tree_node.right)

def categorize(test1_set, dc_tree):
    count_nodes = 0
    for ind, r in test1_set.iterrows():
        res = check_tree(r, dc_tree)
        if res == r['Class']:
            count_nodes = count_nodes + 1
    return count_nodes

# Pruning
## count the nodes and label with a number
def find_nodes_count(root, num_of_nodes, queue = deque()):
    # global num_of_nodes
    if null_check(root):
        return
    if root and type(root.name) is not int:
        num_of_nodes += 1
        root.label = num_of_nodes
    root_child = []
    root_child.append(root.left)
    root_child.append(root.right)
    for node_value in root_child:
        if node_value:
            queue.append(node_value)
    if queue:
        find_nodes_count(queue.popleft(), num_of_nodes, queue)
    else:
        return num_of_nodes

def find_count_non_leaf(root, queue= deque()):
    global non_leaf_nodes, nodes
    if null_check(root):
        return
    if root and type(root.name) != int:
        non_leaf_nodes += 1
        nodes.append(root.name)
    root_child = []
    root_child.append(root.left)
    root_child.append(root.right)
    for node_value in root_child:
        if node_value:
            queue.append(node_value)
    if queue:
        find_count_non_leaf(queue.popleft(), queue)

def prune_node(dc_tree, pos, index):
    if null_check(dc_tree):
        return
    if dc_tree.label == pos:
        dc_tree.name = index
        dc_tree.left = None
        dc_tree.right = None
        dc_tree.label = 0
        return
    if not null_check(dc_tree.left):
        prune_node(dc_tree.left, pos, index)
    if not null_check(dc_tree.right):
        prune_node(dc_tree.right, pos, index)

def divide_tree(root, limit):
    if null_check(root):
        return
    if root.label == limit:
        return root.reflect_x()
    if not null_check(root.left):
        return divide_tree(root.left, limit)
    if not null_check(root.right):
        return divide_tree(root.right, limit)

def calculate_helper(node, zero_cnt, one_cnt):
    if null_check(node):
        return
    if type(node.name) is int:
        if node.name == 0:
            zero_cnt += 1
        else:
            one_cnt += 1
    if not null_check(node.left):
        calculate_helper(node.left, zero_cnt, one_cnt)
    if not null_check(node.right):
        calculate_helper(node.right, zero_cnt, one_cnt)

def iterrandom_num(initial, temp_best):
    global non_leaf_nodes, nodes
    #for the 20 iterations and compute some random number of further iterations
    #These all iterations will give more accuracy. Trying with different posibility to find the position
    #of the node and then classify the class to the node and prune their child nodes. 
    #atmost 600 times will execute
    for j in range(random.randint(initial, 30)):
        num_of_nodes = 0
        #count the number of nodes in given tree
        find_nodes_count(temp_best, num_of_nodes)
        non_leaf_nodes = 0
        nodes = []
        #count non_leaf nodes in the treee
        find_count_non_leaf(temp_best)
        if len(nodes) is 1:
            break
        #position chosen for pruning by difference between the total number of non-leaf nodes and non-leaf nodes/2
        position_gen = random.randint(int(non_leaf_nodes / 2), int(non_leaf_nodes))
        zero_cnt = 0
        one_cnt = 0
        #traverse the tree and move the position chosen in the previous step
        available_nodes = divide_tree(temp_best, position_gen)
        #calculate the number of class 0 and class 1 in available_nodes tree
        calculate_helper(available_nodes, zero_cnt, one_cnt)
        if zero_cnt > one_cnt:
            index = 0
        else:
            index = 1
        #pruned and assigned the class label as the node name   
        prune_node(temp_best, position_gen, index)
    return temp_best

def post_pruning(root, accuracy ):
    initial_best = copy.deepcopy(root)
    #This loop is used to determine better accuracy on 20 different chances instead of single chance
    #Every time best nodes to choosen based on accuracy 
    #This iterations will increase the chance of having good pruning tree based on accuracy
    for i in range(20):
        temp_best = copy.deepcopy(initial_best) 
        tp_best = iterrandom_num(i, temp_best)
        count_at_i = categorize(validation_set, tp_best) #validate with best pruned tree so far to evaluate accuarcy
        temp_acc = float(count_at_i)/float(totalnodes) * 100

        if temp_acc > accuracy:
            initial_best = copy.deepcopy(tp_best) #current prune tree is having better accuracy than prevoius one.So set this next iteration
            accuracy = temp_acc
    return initial_best

# construct a tree with information gain
root = frameNewData(df)
root_variance = frame_data_with_variance(df)

test_count = categorize(test_set, root)
totalnodes = len(test_set['Class'])

entropy_accuracy = (float(test_count)/(float(totalnodes))) * 100
print("H1 NP", entropy_accuracy)
var_tcount = categorize(test_set, root_variance)
variance_accuracy = (float(var_tcount)/float(totalnodes)) * 100
print("H2 NP", variance_accuracy)
##labelling the nodes in the constructed tree
find_nodes_count(root, 0)
find_nodes_count(root_variance, 0)

## to display the tree value based on command line argument
if len(sys.argv) > 4 and sys.argv[4] == 'yes':
    display_tree(root, 0)
    display_tree(root_variance,0)

if len(sys.argv) > 5 and sys.argv[5] == 'yes':
    construct_prune_tree = post_pruning(root, entropy_accuracy)
    categorize_root = categorize(test_set, construct_prune_tree)
    construct_prune_tree_variance = post_pruning(root_variance, variance_accuracy )
    categorize_root_variance = categorize(test_set, construct_prune_tree_variance)
    accuracy1 = float(categorize_root)/float(totalnodes) * 100
    accuracy2 = float(categorize_root_variance)/float(totalnodes) * 100
    if sys.argv[4] == 'yes':
        display_tree(construct_prune_tree, 0)
        display_tree(construct_prune_tree_variance, 0)
    print("H1 P", accuracy1)
    print("H2 P", accuracy2)


