from collections import deque
 
 
# Below lists detail all eight possible movements from a cell
# (top, right, bottom, left)
row = [ 1, -1, 0, 0]
col = [ 0,  0, 1, -1]

 

 
def in_matr_chk(mat, x, y, processed):
    return (x >= 0) and (x < len(processed)) and \
        (y >= 0) and (y < len(processed[0])) and \
        (mat[x][y] == 1 and not processed[x][y])
 
 
def BFS(mat, processed, i, j):
 
    # create an empty queue and enqueue source node
    q = deque()
    q.append((i, j))
 
    # mark source node as processed
    processed[i][j] = True
 
    # loop till queue is empty
    while q:
 
        # dequeue front node and process it
        x, y = q.popleft()
 
        # check for all four possible movements from the current cell
        # and enqueue each valid movement
        for k in range(4):
            # skip if the location is invalid, or already processed, or has water
            if in_matr_chk(mat, x + row[k], y + col[k], processed):
                # skip if the location is invalid, or it is already
                # processed, or consists of water
                processed[x + row[k]][y + col[k]] = True
                q.append((x + row[k], y + col[k]))
 
 
if __name__ == '__main__':
 
    mat = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 1],
        
    ]
 
    (M, N) = (len(mat), len(mat[0]))
 
    # stores if a cell is processed or not
    processed = [[False for x in range(N)] for y in range(M)]
 
    island = 0
    for i in range(M):
        for j in range(N):
            # start BFS from each unprocessed node and increment island count
            if mat[i][j] == 1 and not processed[i][j]:
                BFS(mat, processed, i, j)
                island = island + 1
 
    print("The total number of islands is", island)
 

