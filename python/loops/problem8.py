# Prime number checker

number = 29

its_prime = True

if number>1:
    for i in range(2, number):
        if number%i==0:
            its_prime = False
            break
print("Prime:",its_prime)