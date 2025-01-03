# transportation mode selection - Choose a mode of transportation based on distance
# (e.g. <3Km Walk, 3-15Km: Bike, >15: Car or public)

distance = int(input("Enter distance to travel: "))
if distance<0:
    print("How's this is possible? You are crazy!!")
    exit()
if distance<=3:
    print("You should go by walking.")
elif 3<distance<=15:
    print("You should use your bike.")    
else:
    print("Try to use public transport or car.")
