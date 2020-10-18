#import codecademylib3_seaborn
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-poster')
plt.style.use('seaborn-colorblind')

# Bar Graph: Featured Games

games = ["LoL", "Dota 2", "CS:GO", "DayZ", "HOS", "Isaac", "Shows", "Hearth", "WoT", "Agar.io"]

viewers =  [1070, 472, 302, 239, 210, 171, 170, 90, 86, 71]
ax = plt.subplot()
plt.bar(range(len(games)), viewers, color = 'slateblue', label = 'Twitch')
plt.title('Featured Games Viewers')
plt.xlabel('Games')
plt.ylabel('Viewers')
plt.legend()

ax.set_xticks(range(len(games)))
ax.set_xticklabels(games, rotation = 30)
plt.savefig('bar_chart.png')
plt.show()
plt.close()
# Pie Chart: League of Legends Viewers' Whereabouts

labels = ["US", "DE", "CA", "N/A", "GB", "TR", "BR", "DK", "PL", "BE", "NL", "Others"]

countries = [447, 66, 64, 49, 45, 28, 25, 20, 19, 17, 17, 279]

colors = ['lightskyblue', 'gold', 'lightcoral', 'gainsboro', 'royalblue', 'lightpink', 'darkseagreen', 'sienna', 'khaki', 'gold', 'violet', 'yellowgreen']

explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

plt.pie(countries, explode=explode,colors= colors, shadow = True, startangle =345, autopct ='%1.0f%%', pctdistance = 1.15)
plt.title("League of Legends Viewers' Whereabouts")
plt.legend(labels, loc="right")
plt.show()
plt.savefig('pie_chart.png')
plt.close()
# Line Graph: Time Series Analysis

hour = range(24)

viewers_hour = [30, 17, 34, 29, 19, 14, 3, 2, 4, 9, 5, 48, 62, 58, 40, 51, 69, 55, 76, 81, 102, 120, 71, 63]

ax = plt.subplot()
plt.plot(hour, viewers_hour, label = '2015-01-01')

y_upper = [y+y*0.15 for y in viewers_hour]
y_lower = [y-y*0.15 for y in viewers_hour]

plt.fill_between(hour, y_lower, y_upper, alpha =0.2)
ax.set_xticks(range(len(hour)))
plt.title("Time Series")
plt.xlabel('Hour')
plt.ylabel('Viewers')
plt.legend()
plt.savefig('line_chart.png')
plt.show()
