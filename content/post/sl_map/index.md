---
title: Map
summary: Map
authors:
  - admin
date: "2021-02-18T13:00:00Z"
lastMod: "2021-02-18T00:00:00Z"
#tags: []

# Is this a featured talk? (true/false)
featured: false

#image:
#  caption: ''
#  focal_point: Right
---

```{r, eval = TRUE}
library(leaflet)
leaflet() %>%
  addTiles() %>%
  addMarkers(
    lat = geo_dat$lat,
    lng = geo_dat$long,
    popup = ifelse(geo_dat$z == 0,
                   paste(geo_dat$affil, "<br>",
                         "Researcher:",geo_dat$researcher, "<br>",
                         "<a href=",geo_dat$gs_page,">Google Scholar Page</a>"),
                  ifelse(geo_dat$z == 1,
                                  paste(geo_dat$affil, "<br>",
                                    "Researcher:",geo_dat$researcher, "<br>",
                                    "<a href=",geo_dat$gs_page,">Google Scholar Page</a>","<br>",
                                    "Researcher:",geo_dat$researcher2, "<br>",
                                    "<a href=",geo_dat$gs_page2,">Google Scholar Page</a>"),
                        paste(geo_dat$affil, "<br>",
                        "Researcher:",geo_dat$researcher, "<br>",
                        "<a href=",geo_dat$gs_page,">Google Scholar Page</a>","<br>",
                        "Researcher:",geo_dat$researcher2, "<br>",
                        "<a href=",geo_dat$gs_page2,">Google Scholar Page</a>","<br>",
                        "Researcher:",geo_dat$researcher3, "<br>",
                        "<a href=",geo_dat$gs_page3,">Google Scholar Page</a>")

    )))

```
