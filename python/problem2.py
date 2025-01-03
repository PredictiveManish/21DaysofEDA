# Movie ticket pricing

days = input("Enter the day of week: ")
age = int(input("Enter your age: "))
if days=='Wednesday':
    print("You will get $2 discount.")  
    if(age<18):
        print("Your ticket price is $6(Child Original-$8).")
    else:
        print("You have to pay $10(Adult Original-12$).")
else:
    if(age<18):
        print("Your ticket price is $8.")
    else:
        print("You have to pay $12.")