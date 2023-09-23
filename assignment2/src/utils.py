from itertools import chain, combinations
from efficient_apriori import apriori


def powerset(iterable, min_len=None, max_len=None):
    s = list(iterable)
    if min_len is None:
        min_len = 0
    if max_len is None:
        max_len = len(s)
    return chain.from_iterable(combinations(sorted(s), r) for r in range(min_len, max_len+1))


def unique_items(transactions):
    unique_items = set()
    for t in transactions:
        for item in t:
            unique_items.add(item)
    return unique_items


def merge_itemsets(a, b):
    itemset = set(a).union(set(b))
    itemset = tuple(sorted(itemset))
    return itemset


def support_count(transactions, itemset):
    # Set the initial support count to 0
    support_count = 0
    # Check for each transaction if it contains the itemset
    # If so, increment support count
    for t in transactions:
        if set(itemset).issubset(set(t)):
            support_count += 1
    # Return support count
    return support_count


def support(transactions, itemset):
    if len(transactions) == 0:
        return 0.0
    # Return support count
    return support_count(transactions, itemset) / len(transactions)


def confidence(transactions, association_rule):
    # Split association rule into itemsets X and Y (reflecting X=>Y)
    X, Y = association_rule
    # Calculate the support count for X
    support_count_X = support_count(transactions, X)
    # If the support count of X is 0, return 0 to avoid division by zero
    if support_count_X == 0:
        return 0.0
    # Calculate X*union*Y
    itemset = tuple(sorted(set(X).union(set(Y))))
    # Caluculate and return the confidence
    return support_count(transactions, itemset) / support_count_X


def generate_association_rules(itemset):
    assoc_rules = []
    for X in powerset(itemset, min_len=1, max_len=len(itemset)-1):
        Y = tuple(sorted(set(itemset).difference(set(X))))
        assoc_rules.append((X, Y))
    return assoc_rules




def show_top_rules(transactions, min_support=0.0, min_confidence=0.0, k=5, sort='lift', rhs=None):

    _, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)

    rule_count = len(rules)
    
    # Do the filtering
    if rhs is not None:
        rules = [ r for r in rules if len(r.rhs) == 1 and r.rhs[0] == rhs ]

    rule_count_filtered = len(rules)    
        
    # Do the sorting
    if sort == 'support':
        rules = sorted(rules, key=lambda rule: rule.support, reverse=True)
    elif sort == 'confidence':
        rules = sorted(rules, key=lambda rule: rule.confidence, reverse=True)
    else:
        rules = sorted(rules, key=lambda rule: rule.lift, reverse=True)
    
    
    # Do the capping
    rules = rules[0:(k+1)]
    
    # Do the showing
    print('=== Total Number of Rules: {} | Number of rules with matching RHS: {} ==='.format(rule_count, rule_count_filtered))
    for r in rules:
        print('{} => {}: supp: {:.3f}, conf: {:.3f}, lift: {:.3f}'.format(r.lhs, r.rhs, r.support, r.confidence, r.lift))
    print()