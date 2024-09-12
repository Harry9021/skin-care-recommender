import React from "react";
import "../Styles/Navbar.css";
import DownArrow from "../Vectors/downarrow.svg"
import Cart from "../Vectors/cart.png"

export default function Navbar() {
    return (
        <>
            <div className="navbar">
                <div className="logger-black"></div>
                <div className="logo">SKINCARE</div>
                <div className="logger">
                    <div className="logger-1">
                        <div className="loginButton">
                            <div style={{ fontSizes: 20 }}>Login/Signup</div>
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
        </>
    );
}
