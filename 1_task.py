
def sum_for(N):
    Sum = 0
    for i in range(N+1):
       Sum += i
    return Sum
 
def sum_formula(N):
    Sum=int((N+1)*N*0.5)
    return Sum
 
    
def main():
   
    N = int(input('plese txt integer number: '))
    if N<=10**25:
        print('sum_for',sum_for(N))
        print('sum_formula = ',sum_formula(N))
    else:
        print('limits exceeded')
if __name__ == '__main__':
       
    main()