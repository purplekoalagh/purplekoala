import torch

def rowswap(matrix: torch.tensor, source_row_index: int, target_row_index: int) -> torch.tensor:
  """
  Swaps rows based on row indexes
  The indexes should start at 0 and end at n-1 where n is the number of rows

  Formula: R1 <-> R2

  Args:
      matrix (torch.tensor): a torch tensor that contains a matrix
      source_row_index (int): an integer value that contains the source row's index
      target_row_index (int): an integer value that contains the target row's index

  Returns:
      (torch.tensor): a torch tensor that contains a matrix

  Example:
      matrix = torch.tensor([[ 1,3,0,0,3], [0,0,1,0,9],[0,0,0,1,-4]], dtype=torch.float16)
      [1,3,0,0,3]
      [0,0,1,0,9]
      [0,0,0,1,-4]
      >>>rowswap(matrix, 0, 1)
      [0,0,1,0,9]
      [1,3,0,0,3]
      [0,0,0,1,-4]
  """


  temp = matrix[source_row_index,:].clone() #uses clone since slicing reutrns a link to the orginal matrix meaning if changed then this variable will change also
  matrix[source_row_index,:] = matrix[target_row_index,:]
  matrix[target_row_index,:] = temp

  return matrix

def rowscale(matrix: torch.tensor, source_row_index: int, scaling_factor: float) -> torch.tensor:
  """
  Scales a row based on row index and value

  Formula: R1 = R1 * 2

  Args:
      matrix (torch.tensor): a torch tensor that contains a matrix
      source_row_index (int): an integer value that contains the source row's index
      scaling_factor (float): a float value that contins the scaling factor

  Returns:
      (torch.tensor): a torch tensor that contains a matrix

  Example:
      matrix = torch.tensor([[ 1,3,0,0,3], [0,0,1,0,9],[0,0,0,1,-4]], dtype=torch.float16)
      [1,3,0,0,3]
      [0,0,1,0,9]
      [0,0,0,1,-4]
      >>>rowscale(matrix, 0, 2)
      [2,6,0,0,6]
      [0,0,1,0,9]
      [0,0,0,1,-4]
  """

  """
  rows, columns = matrix.size()
  matrix[source_row_index,:] = matrix[source_row_index,:] * scaling_factor
  """

  matrix_numpy = matrix.numpy()
  rows, columns = matrix_numpy.shape
  for i in range(0,columns):
    matrix[source_row_index][i] = scaling_factor * matrix[source_row_index][i]
  return matrix

def rowreplacement(matrix: torch.tensor, first_row: torch.tensor, second_row: torch.tensor, j: float, k: float) -> torch.tensor:
  """
  Replaces a row by combining scalar multiples of itself and another row

  Formula R1 = jR1 + kR2

  Args:
      matrix (torch.tensor): a torch tensor that contains a matrix
      first_row (torch.tensor): a torch tensor that contains the row to be replaced
      second_row (torch.tensor): a torch tensor that contains the second row
      j (float): scalar value that is multiplied with the first row entered
      k (float): scalar value that is multipled with the second row 
       
  Returns:
      torch.tensor: a torch tensor that contains a matrix
      
  Example:
      matrix = torch.tensor([[ 1,3,0,0,3], [0,0,1,0,9],[0,0,0,1,-4]], dtype=torch.float16)
      rowreplacement(matrix, matrix[0,:], matrix[1,:], 1, 2)
      [1, 3, 2, 0,21]
      [0, 0, 1, 0, 9]
      [0, 0, 0, 1,-4]
  """

  replacement_row = first_row * j
  replacement_row = replacement_row.add(second_row*k)

  rows, columns = matrix.size()

  for index in range(0,rows):
    if (torch.equal(matrix[index,:],first_row)):
      first_row_index = index
      break
  matrix[index,:] = replacement_row

  return matrix

def rref(matrix: torch.tensor) -> torch.tensor:
  """
  Transforms a torch.tensor matrix into its reduced row echelon form

  Args:
      matrix (torch.tensor): a torch.tensor that contians a matrix
      
  Returns:
      (torch.tensor): a torch.tensor that contians a matrix
      
  Example:
      matrix = torch.tensor([[ 1,3,0,0,3], [0,0,1,0,9],[0,0,0,1,-4], [4,4,4,4,4]], dtype=torch.float16)
      >>>rref(matrix)
      [1.0000,  1.0000,  1.0000,  1.0000,  1.0000]
      [0.0000,  1.0000, -0.4995, -0.4995,  1.0000]
      [0.0000,  0.0000,  1.0000,  0.0000,  9.0000]
      [0.0000,  0.0000,  0.0000,  1.0000, -4.0000]           
  """

  #determine size
  rows, columns = matrix.size()

  # Places a one in pivot spot
  row_that_needs_one = 0
  for column_index in range(0, columns):
    for row_index in range(0, rows):
      if (matrix[row_index, column_index] == 1):
        rowswap(matrix, row_index, row_that_needs_one)
        row_that_needs_one +=1
        row_that_needs_one = row_that_needs_one%rows
        break

  # Get Pivot Indexes
  pivot_index = []
  for column_index in range(0, columns):
    for row_index in range(0, rows):
      if (matrix[row_index, column_index] == 1):
        pivot_index.append((row_index,column_index))

  # Append Additional Pivots if necessary (i.e. were not one to begin)
  if rows != len(pivot_index):
    row_list = [x[0] for x in pivot_index]
    column_list = [x[1] for x in pivot_index]

    # determine missing pivot indexes
    for row_index in range(0, rows):
        if row_index not in row_list:
          for column_index in range(0, len(column_list)):
            if column_index not in column_list:
                pivot_index.append((row_index,column_index))

    for index in pivot_index:
      if(matrix[index].numpy() != 1 and matrix[index].numpy() != 0):
        matrix = rowscale(matrix, index[0], (1/matrix[index]))

    # Places a number in pivot spot
    row_that_needs_one = 0
    for column_index in range(0, columns):
      for row_index in range(0, rows):
        if (matrix[row_index, column_index] != 0):
          rowswap(matrix, row_index, row_that_needs_one)
          row_that_needs_one +=1
          row_that_needs_one = row_that_needs_one%rows
          break

    # Get new pivots
    pivot_index = []
    last_col = 0
    for row_index in range(0,rows):
      for column_index in range(last_col, columns):
        if (matrix[row_index, column_index] != 0):
          pivot_index.append((row_index,column_index))
          last_col = column_index +1
          break

  # Make all pivots 1 if not
  for index in pivot_index:
    rowscale(matrix, index[0], (1/matrix[index]))

  # Make all below elements 0
  for index in pivot_index:
    for row_index in range(index[0]+1, rows-1):
      if matrix[row_index,index[1]].item() != 0:
        if matrix[row_index,index[1]] < 0:
          rowreplacement(matrix,matrix[row_index,:],matrix[index[0],:],1,(1/matrix[row_index,index[1]]))
        else:
          if matrix[row_index,index[1]] < 1:
            rowreplacement(matrix,matrix[row_index,:],matrix[index[0],:],1.0,(-matrix[row_index,index[1]]))
          else:
            rowreplacement(matrix,matrix[row_index,:],matrix[index[0],:],1.0,(-1/matrix[row_index,index[1]]))

  # Make all pivots 1 if not
  for index in pivot_index:
    rowscale(matrix, index[0], (1/matrix[index]))

  return matrix

'''matrix = torch.tensor([[0,0,0,1,-4],[0,0,1,0,9],[ 1,3,0,0,3], [4,4,4,4,4]], dtype=torch.float16)
print(matrix)
print("------------------------------")
rref(matrix)'''

if __name__ == "__main__":
  matrix = torch.tensor([[ 1,3,0,0,3], [0,0,1,0,9],[0,0,0,1,-4]], dtype=torch.float16)
  print("Matrix before rowswap", matrix)
  rowswap(matrix, 0, 1)
  print("Matrix after rowswap", matrix)
  rowscale(matrix, 0, (1/3))
  print("Matrix after rowscale", matrix)
  rowreplacement(matrix, matrix[2,:], matrix[0,:], 1, -3)
  print("Matrix after rowreplacement", matrix)
  rref(matrix)
  print(matrix)

