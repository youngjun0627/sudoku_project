def nine_by_nine(grid):
    if not grid:
        return False
    if len(grid) != 9:
        return False
    for row in grid:
        if len(row) != 9:
            return False
    return True    

def all_digits(grid):
    for row in grid:
        for cell in row:
            try:
                int_value = int(cell)
                if int_value < 0 or int_value > 9 or int_value != cell:
                    return False
            except:
                return False
    return True   


# checks no duplicates from 1 to 9 (duplicate 0's are ok)
def check_no_dups(nineGrouping):  
    uniqueDigits = set()
    for num in nineGrouping:
      if num != 0:
        if num  in uniqueDigits:
           return False
      uniqueDigits.add(num)
    return True
    

def check_row(grid, row):
    nineGrouping = []
    for cell in grid[row]:
        nineGrouping.append(cell)
    return check_no_dups(nineGrouping)

def check_column(grid, column):
    nineGrouping = []
    for row in grid:
        nineGrouping.append(row[column])
    return check_no_dups(nineGrouping) 

def check_subgrid(grid, row, column):
    nineGrouping = []
    for i in range(row, row + 3):
        for j in range(column, column + 3):
            nineGrouping.append(grid[i][j])
    return check_no_dups(nineGrouping)

def check_sudoku(grid):
    if not nine_by_nine(grid) or not all_digits(grid):
      return None
    for i in range(0, 9):
      if not check_row(grid, i):
          return False
      if not check_column(grid, i):
	      return False
    for i in range(0, 7, 3):
      for j in range(0, 7, 3):
        if not check_subgrid(grid, i, j):
            return False

    return True
