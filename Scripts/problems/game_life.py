import numpy as np

DIED_CELL = 0
LIVE_CELL = 1
OUT_OF_BORDER_VALUE = 0

AXIS_X = 0
AXIS_Y = 1


def init_table(zero_state=[], table_size=10):
    table = np.tile([DIED_CELL], (table_size, table_size))
    for cell in zero_state:
        table[cell[AXIS_X], cell[AXIS_Y]] = LIVE_CELL
    return table

def get_next_cell_state(cell_state, neighbors):
    living_neighbors = np.sum(neighbors)
    if cell_state == 1:
        is_cell_still_living = living_neighbors == 2 or living_neighbors == 3
        if not is_cell_still_living:
            next_value = 0
        else:
            next_value = 1
    if cell_state == 0:
        is_a_birth_cell = living_neighbors == 3
        if is_a_birth_cell:
            next_value = 1
        else:
            next_value = 0
    return next_value

def get_next_state(table):
    neighbors_shift = np.array([[-1,1], [0,1], [1,1], [-1,-1], [0,-1], [1,-1], [-1,0], [1,0]])
    next_state = init_table([], len(table))
    for x, row in enumerate(table):
        for y, cell_state in enumerate(row):
            cell_position = np.array([x, y])
            neighbors_position = neighbors_shift + cell_position
            neighbors_value = []
            for neighbor_position in neighbors_position:
                try:
                    neighbors_value.append(table[neighbor_position[AXIS_X]][neighbor_position[AXIS_Y]])
                except Exception:
                    neighbors_value.append(OUT_OF_BORDER_VALUE)
            next_cell_state = get_next_cell_state(cell_state, neighbors_value)
            next_state[cell_position[AXIS_X]][cell_position[AXIS_Y]] = next_cell_state
    return next_state

def get_next_states(table, steps=2):
    if steps == 1:
        return get_next_state(table)
    else:
        following_state = get_next_state(table)
        print(following_state)
        return get_next_states(following_state, steps=steps-1)



def main():
    # Sparse coordinates
    zero_state = [(1,0), (0,1), (1,1), (2,1), (1,2)]

    table_init = init_table(zero_state, table_size=4)
    print(table_init)
    next_table = get_next_states(table_init, 2)
    print(next_table)


main()