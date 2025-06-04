import turtle
import random
import math

odd_number_as_sqrt_of_population = 15

infection_prob = 50

recovery_chance = 50

population = odd_number_as_sqrt_of_population**2

recovery_time = 100

social_distancing = False
start_sd = 30

sd_p = 100

quarantaine_on = True
time_until_q = 3000


wn = turtle.Screen()
wn.bgcolor('black')
wn.title('Epidemic simulator')
wn.tracer(0)

pen = turtle.Turtle()
pen.speed(0)
pen.color('white')
pen.penup()
pen.setposition(-200,-200)
pen.pendown()
pen.pensize(3)
for i in range(4):
    pen.fd(400)
    pen.lt(90)
pen.hideturtle()

time = 0
balls = []

infected = []

time_infected = {}
tot_inf = []

recovered = []

closest_ball = []
spaces_x = []
spaces_y = []
coordinates = []

for i in range(-200,201):
    if i%int(400/math.sqrt(population)) == 0: #this is a calculation I created to make sure that more coordinates are created with a larger population
        
        spaces_x.append(i)
        spaces_y.append(i)

for x in spaces_x:
    for y in spaces_y:
        coordinate = [x,y]
        coordinates.append(coordinate)

for _ in range(population):
    balls.append(turtle.Turtle())

for ball in balls:
    ball.shape('circle')
    ball.shapesize(0.45,0.45)
    ball.color('#4183C4')
    ball.penup()
    ball.speed(0)
    x = random.randint(-190, 190)
    y = random.randint(-190, 190)
    ball.goto(x, y)
    ball.dx = random.choice([-3,-2.75,-2.5,-2.25,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,3,2.75,2.5,2.25,2,1.75,1.5,1.25,1,0.75,0.5,0.25])
    ball.dy = random.choice([-3,-2.75,-2.5,-2.25,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,3,2.75,2.5,2.25,2,1.75,1.5,1.25,1,0.75,0.5,0.25])

infections = {i : 0 for i in balls}


def closest_to_O():
    global closest_ball

    min_dist = 0
    for x in range(len(balls)):

            xdist = balls[x].xcor()
            ydist = balls[x].ycor()
            dist_squared = xdist*xdist + ydist*ydist
            if len(closest_ball) == 0:
                closest_ball.append(balls[x])
                min_dist = dist_squared
            elif dist_squared < min_dist:
                closest_ball.clear()
                closest_ball.append(balls[x])
                min_dist = dist_squared


def mark_infected():
    global closest_ball
    
    if closest_ball[0] not in recovered:
        closest_ball[0].color('red')
        infected.append(closest_ball[0])


def collide():
    for i in range(len(balls)):
        for x in range(i + 1, len(balls)):
            ball1 = balls[i]
            ball2 = balls[x]

            xdist = ball1.xcor() - ball2.xcor()
            ydist = ball1.ycor() - ball2.ycor()
            dist_squared = xdist * xdist + ydist * ydist

            if dist_squared < 125:
                if ball1 in recovered or ball2 in recovered:
                    continue

                if ball1 in infected and ball2 not in infected:
                    if random.randint(0, 100) <= infection_prob:
                        ball2.color('red')
                        infected.append(ball2)
                    elif random.randint(0, 100) <= recovery_chance:
                        ball1.color('gray')
                        if ball1 not in recovered:
                            recovered.append(ball1)

                # ball2 infected, ball1 healthy
                elif ball2 in infected and ball1 not in infected:
                    if random.randint(0, 100) <= infection_prob:
                        ball1.color('red')
                        infected.append(ball1)
                    elif random.randint(0, 100) <= recovery_chance:
                        ball2.color('gray')
                        if ball2 not in recovered:
                            recovered.append(ball2)


def recovered_():

    global population, recovery_time

    for i in range(len(balls)):
        if balls[i] in infected:
            infections[balls[i]] += 1
            if balls[i] not in recovered:
                if infections[balls[i]] >= recovery_time:
                    balls[i].color('gray')
                    recovered.append(balls[i])

def social_dist():

    min_dist = 0
    closest_dist = []

    for i in range(len(balls)):

        if balls[i].xcor() > -250:
            if random.randint(0,100) in range(0, sd_p + 1):

                for c in coordinates:

                    xdist = balls[i].xcor() - c[0]
                    ydist = balls[i].ycor() - c[1]
                    dist_squared = xdist*xdist + ydist*ydist
                    if len(closest_dist) == 0:
                        closest_dist = c
                        min_dist = dist_squared
                    elif dist_squared < min_dist:
                        closest_dist = c
                        min_dist = dist_squared
            
                if len(coordinates) > 0:
                    try:
                        coordinates.remove(closest_dist)
                    except:
                        coordinates.remove(coordinates[0])
                    balls[i].dx = 0
                    balls[i].dy = 0
                    balls[i].goto(closest_dist[0], closest_dist[1])

                closest_dist.clear()

def quarantaine():
    global time_infected    
    
    
    room = turtle.Turtle()
    room.speed(0)
    room.color('white')
    room.penup()
    room.setposition(-360,-210)
    room.pendown()
    room.pensize(3)
    for i in range(4):
        room.fd(120)
        room.lt(90)
    room.hideturtle()

    for i in infected:
        if i not in tot_inf:
            tot_inf.append(i)

        elif i.xcor() >= -350 and i.xcor() <= -250 and i.ycor() >= -200 and i.ycor() <= -100:
            i.dx = 0
            i.dy = 0
            i.setposition(random.randint(-350,-250), random.randint(-200,-100))
            time_infected[i] = 0

        elif i not in time_infected.keys():
            time_infected[i] = time
        elif i in time_infected.keys():
            if time - time_infected[i] >= time_until_q and i.xcor() > -250:
                dx_q = (random.randint(-275,-225) - i.xcor())*(-1)
                dy_q = random.randint(-175,-125) - i.ycor()
                try:
                    rc = dy_q/dx_q
                    if -4 < rc < 4:
                        i.dx = -10
                        i.dy = 10 * rc
                    else:
                        i.dx = 0
                        i.dy = 0
                        i.goto(random.randint(-350,-250), random.randint(-200,-100))
                        time_infected[i] = 0
                except:
                    i.dx = 0
                    i.dy = 0
                    i.goto(random.randint(-350,-250), random.randint(-200,-100))
                    time_infected[i] = 0
                time_infected[i] = 0
            
    

closest_to_O()
mark_infected()

while True:
        
    wn.update()

    for ball in balls:
        
        time += 1

        ball.sety(ball.ycor() + ball.dy)
        ball.setx(ball.xcor() + ball.dx)

        if ball.xcor() < -195 or ball.xcor() > 195:
            ball.dx *= -1
            
        if ball.ycor() < -195 or ball.ycor() > 195:
            ball.dy *= -1

    collide()
    recovered_()

    if social_distancing and len(tot_inf) >= start_sd:
        social_dist()

    if quarantaine_on:
        quarantaine()

    for i in infected: 
        if i not in tot_inf: 
            tot_inf.append(i)
    

    if len(recovered) < len(tot_inf):
        print('\n', 'Percentage infected:', len(tot_inf)/population * 100, '\n',
        'Infected people:' , len(tot_inf), '\n' , 'Recovered people:', len(recovered), '\n', 'Time:' , time , 'minutes')
    if len(recovered) == len(tot_inf) and len(recovered) > 0:
        message = 'Percentage infected: '+str(len(tot_inf)/population * 100)+'\n'+'Infected people: '+str(len(tot_inf))+'\n'+'Recovered people: '+str(len(recovered))+'\n'+'Time: '+str(time)+' minutes'
        end_message = turtle.Turtle()
        end_message.speed(0)
        end_message.color('white')
        end_message.penup()
        end_message.setposition(-190,205)
        end_message.pendown()
        end_message.pensize(10)
        end_message.write(message, font = ('style', 18, 'bold'))
        end_message.hideturtle()
        break

wn.mainloop()