import React from "react";
import "../../Styles/Results.css";

function Resultcard({ item, recommendationNumber, addToCart }) {
    return (
        <div className="res-card">
            <div className="res-img">{recommendationNumber}.</div> {/* Replace with actual image later */}
            <div className="res-details">
                <div className="res-details-upper">
                    <div className="res-brand capitalize">{item.label}</div>
                    <div className="res-name">
                        <span className="res-name-span capitalize" title={item.name}>{item.name}</span>
                    </div>
                </div>
                <div className="res-details-lower">
                    <div className="res-price">₹ {item.price}</div>
                    {/* Add to Cart Button */}
                    <div className="add-to-cart" onClick={() => addToCart(item)}>Add to Cart</div> 
                </div>
            </div>
        </div>
    );
}

export default Resultcard;
