# Grade Calculator
marks = int(input("Enter your marks in the subject: "))
if marks>100:
    print("Please enter valid input.")
    exit()
if(marks>=90):
    grade = "A"
elif(89<=marks<=80):
    grade = "B"
elif(70<=marks<=79):
    grade = "C"
elif(60<=marks<=69):
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")

