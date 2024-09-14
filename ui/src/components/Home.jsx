import React from "react";
import { Link } from "react-router-dom";
import "../Styles/Home.css";
import "../Styles/Navbar.css";
import DownArrow from "../Vectors/downarrow.svg";
import Cart from "../Vectors/cart.png";
import image1 from "../Vectors/image1.png";
import image2 from "../Vectors/image2.png";
import image3 from "../Vectors/image3.png";
import image4 from "../Vectors/image4.png";

export default function Home() {
    return (
        <div className="home">
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
                    <Link to="/cart">
                        <div className="cart">
                            <img src={Cart} alt="" />
                        </div>
                    </Link>
                </div>
            </div>
            <div className="home-content">
                <div className="imager">
                    <img className="image-1" src={image1} alt="" />
                    <img className="image-2" src={image2} alt="" />
                    <img className="image-3" src={image3} alt="" />
                    <img className="image-4" src={image4} alt="" />
                </div>
                <div className="written-content">
                    <div className="written">
                        Here’s How You Can Determine Your Skin Type . Knowing
                        your skin type is essential for creating a skincare
                        routine that works for you.
                    </div>
                    <div className="try-button-div">
                        <Link to="/form">
                            <button className="try-button">Try Now</button>{" "}
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
}
