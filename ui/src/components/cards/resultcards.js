import React from 'react'
import "../../Styles/Results.css"

function resultcards() {

  const data=[{"label":"face-moisturisers","brand":"Garnier","name":"Skin Naturals BB Cream 30 g","price":140}];

  return (
    <>
      <div className="res-card">
        <div className="res-img"></div>
        <div className="res-details">
          <div className="res-details-upper">
            <div className="res-brand"></div>
            <div className="res-name">{data[0].name}</div>
          </div>
          <div className="res-details-lower">
            <div className="res-price"></div>
            <div className="add-to-cart"></div>
          </div>
        </div>
      </div>
    </>
  )
}

export default resultcards