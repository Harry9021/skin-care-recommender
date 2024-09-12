import React from "react";
import Navbar from "./Navbar";
import "../Styles/Home.css";
import image1 from "../Vectors/image1.png";
import image2 from "../Vectors/image2.png";
import image3 from "../Vectors/image3.png";
import image4 from "../Vectors/image4.png";

export default function Home() {
    return (
        <div className="home">
            <Navbar />
            <div className="home-content">
                <div className="imager">
                    <img className="image-1"src={image1} alt="" />
                    <img className="image-2"src={image2} alt="" />
                    <img className="image-3"src={image3} alt="" />
                    <img className="image-4"src={image4} alt="" />
                </div>
                <div className="written-content">
                    <div className="written">
                        Here’s How You Can Determine Your Skin Type . Knowing
                        your skin type is essential for creating a skincare
                        routine that works for you.
                    </div>
                    <div className="try-button-div">
                        <button className="try-button">Try Now</button>
                    </div>
                </div>
            </div>
        </div>
    );
}
