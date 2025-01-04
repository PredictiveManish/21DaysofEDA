# factorial using while loop

number = int(input("Enter any number: "))
factorial = 1

while number>0:
    factorial = factorial * number
    number-=1
print(factorial)