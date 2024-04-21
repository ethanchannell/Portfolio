# Warehouse Distribution Center Labor Cost Saving Strategies
This project was completed for the Senior Design Competition at Georgia Tech for Industrial Engineers. This is a year long competition where a team of students connect with a company as a client and completes a project for them using the skills learned from the Industrial Engineering curriculum. You can read more specific details about the competition in general [here](https://www.isye.gatech.edu/academics/bachelors/current-students/senior-design).

# Introduction
For this project, our team worked with NAPA Autoparts as our client with a specific focus on analyzing their newly built Nashville Distribution Center to see how we could improve operations. I was responsible for creating a tool that could be used by NAPA Autoparts that would lead to reduce labor cost from their putwall within the warehouse. The putwall for NAPA was a term used to describe their sorting system. Essentially, stores are assigned groups on the putwall and the pick and sorting for them would be done in waves. With NAPA's current system the putwall assignments were static where the store groups wouldn't be evaulated until months at a time passed. The tool I ended up creating allowed for a dynamic system instead where the putwall assignments would change daily to save on labor cost.

# The Dynamic Putwall Tool
The tool I created for the putwall was created in python and can be seen below.

![image](https://user-images.githubusercontent.com/42851869/148291358-415ebeba-0b6f-4160-94a8-07a77799edd6.png)

Breakingdown of the tool:
1. The user uploads the previous putwall assignments and the orders to be picked today.
2. This determines the number of putwalls to use. The user can either manually choose the number of putwalls or choosing the dynamic option which will choose the number of putwalls based on number of stores ordering and volume.
3. This determines the number of minimum and maximum number of orderlines assigned to each putwall
4. The number of iterations run by the program to find a solution. The more iterations the better the solution.
5. The maximum number of stores switching putwalls so in the case of 60 a max of 60 stores would be assigned new putwalls.

Once these are all filled out the user would press assign the program would run and produce files that display the new assignments for the putwalls and expected savings.

# Overview of The Tool
The tool at its core balances the costs of labor associated with changing the putwall with the expected savings in putwall labor and labor in picking the items in the warehouse. Calculating the cost of labor for changing the putwall is pretty easy to do since there is a flat time cost of changing a stores location. Similiarly calculating the savings of using more or less putwalls was also pretty straightforward in calculating and easy to get. The challenging savings to calculate is the how much time is saved for the new picking routes. In order to calculate this, the program had to simulate the new distance the employees would have to travel in the warehouse to pick all the stores items on the putwalls. Once the savings and costs are determined, the tool is able to produce a solution that will guarentee that labor cost is saved if the putwall is changed in any fashion.

# Indepth View of the Tool
The tool created is fully built in python. It can be broken down in the following steps below:
1. Recoginizing the current putwall setup, orderlines and item locations.
2. Building the function that simulates a picker's path within the warehouse given the putwall.
3. Building the function that generates new solutions for the putwall
4. Creating various output files that displays the new putwall setups and expected savings.

For the first step of the tool, the uploaded files are checked and ensured that they are in expected format. If these files are not in the expected format then the tool will provide and error and let the user know that another file format is needed.

The next step involves in simulating a picker's path given the orderlines and putwall setup. In this stage the function determines the distance the picker travels in feet between each item picked in their path and sums it together to give the total distance to collect all the items for the putwall. This function when being built was compared with true distance and travel distance and was highly accurate.

The third step works by building new solutions for the putwall. To build new solutions, I used the genetic algorithm to generate new solutions. The genetic algorithm was utilized because the program has a limited amount of time to run and when attempting to solve the problem using other methods such as Google OR tools VRP package the time to solve was far too long. The genetic algorithm was setup so that the distance traveled for picking, the number of putwalls utilizes, and number of moves for moving stores on the putwall were accounted for in the cost function. In addition, there were high penalty costs associated with having too much order volume on a putwall or having too many stores on a putwall to ensure that these solutions would not be picked as an improvement.

The final step once the final putwall is decided was to take the newly assigned putwalls and export them as csv and image file detailing the new putwall assignments and order bactches for each putwall. In addition, the estimated time for implementing the new putwalls would also be displayed so that NAPA would know their expected savings from completing these moves.



