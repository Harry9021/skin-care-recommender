import React, { useState } from "react";
import "../../Styles/Results.css";

function Resultcard({ item, recommendationNumber, addToCart }) {
    const [itemAdded, setItemAdded] = useState(false);

    const handleAddToCart = () => {
        addToCart(item);  // Call the function to add the item to the cart
        setItemAdded(true); // Set itemAdded to true to indicate the item is added
    };

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
                    <div
                        className={itemAdded ? "added-to-cart" : "add-to-cart"}
                        onClick={!itemAdded ? handleAddToCart : null}
                    >
                        {itemAdded ? "Added to Cart" : "Add to Cart"}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Resultcard;
