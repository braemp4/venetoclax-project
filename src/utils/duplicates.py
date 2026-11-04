def get_unique_ids(id_list):    
    appearance = {}
    
    uniques = []

    for idx in id_list:
        if idx not in appearance:
            appearance[idx] = 0
        elif idx in appearance:
            appearance[idx] +=1

        uniques.append(f'{idx}_{appearance[idx]}')


    return uniques
