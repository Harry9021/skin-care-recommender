import React from "react";
import "../Styles/Results.css";
import DownArrow from "../Vectors/downarrow.svg";
import Cart from "../Vectors/cart.png";
import { useLocation } from "react-router-dom";
import Resultcard from "./cards/Resultcard";

export default function Results() {
    const location = useLocation();
    const recommendations = location.state?.recommendations || {}; // Default to an empty object

    // Check if recommendations is an object with data
    let items = [];
    if (recommendations.data) {
        try {
            items = JSON.parse(recommendations.data); // Parse the JSON string
        } catch (error) {
            console.error("Error parsing recommendations data:", error);
        }
    }

    return (
        <div className="results">
            <div className="navbar">
                <div className="logger-black">Here are Suggestions for you!</div>
                <div className="logger">
                    <div className="logger-1">
                        <div className="loginButton">
                            <div style={{ fontSize: 20 }}>Login/Signup</div>
                            <div style={{ fontSize: 24 }}>My Account</div>
                        </div>
                        <div className="arrow">
                            <img src={DownArrow} alt="" />
                        </div>
                    </div>
                    <div className="cart">
                        <img src={Cart} alt="" />
                    </div>
                </div>
            </div>
            <div className="result-content">
                {items.length > 0 ? (
                    items.map((item, index) => (
                        <Resultcard key={index} item={item} recommendationNumber={index + 1} />
                    ))
                ) : (
                    <div>No recommendations available.</div>
                )}
            </div>
        </div>
    );
}
