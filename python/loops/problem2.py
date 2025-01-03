# Sum of even numbers
sum = 0
n=int(input("Enter any number: "))
for i in range(1,n+1):
    if i%2==0:
        sum+=i
print(f"Total sum upto {n}: {sum}")