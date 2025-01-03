# Sum of even numbers
sum = 0
n=int(input("Enter any number: "))
for i in range(1,n):
    if i%2==0:
        sum+=i
    else:
        continue
print(f"Total sum upto {n}: {sum}")