import React, { useState } from "react";
import "../Styles/Formpage.css";
import image1 from "../Vectors/image5.png";
import image2 from "../Vectors/image6.png";
import { Link, useNavigate } from "react-router-dom";
import { IoMdArrowRoundBack } from "react-icons/io";

const Formpage = () => {
    const [formData, setFormData] = useState({
        skinType: "",
        concern1: "",
        concern2: "",
        concern3: "",
    });

    const navigate = useNavigate();

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value,
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        const userInput = {
            skin_type: getSkinTypeValue(formData.skinType),
            concern_1: getConcernValue(formData.concern1),
            concern_2: getConcernValue(formData.concern2),
            concern_3: getConcernValue(formData.concern3),
        };

        try {
            const response = await fetch(
                "http://localhost:4000/get-recommendations",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify(userInput),
                }
            );

            const recommendations = await response.json();
            console.log("Recommended Products:", recommendations);
            navigate("/results", { state: { recommendations } });

        } catch (error) {
            console.error("Error fetching recommendations:", error);
            alert("Failed to retrieve recommendations. Please try again.");
        }
    };

    const getSkinTypeValue = (skinType) => {
        switch (skinType) {
            case "dry": return 0;
            case "oily": return 1;
            case "combination": return 2;
            case "normal": return 3;
            default: return 0;
        }
    };

    const getConcernValue = (concern) => {
        const concernsMapping = {
            "anti-pollution": 0, "tan-removal": 1, "dryness": 2,
            "deep-nourishment": 3, "blackheads": 4, "oil-control": 5,
            "fine-lines": 6, "uneven-skin-tone": 7, "dark-spots": 8,
            "dark-circles": 9, "skin-tightening": 10, "under-eye": 11,
            "skin-inflammation": 12, "general-care": 13, "redness": 14,
            "skin-sagging": 15, "lightening": 16, "sun-protection": 17,
            "pigmentation": 18, "blackheads-removal": 19, "oily-skin": 20,
            "anti-ageing": 21, "hydration": 22, "dull-skin": 23,
            "uneven-texture": 24, "irregular-textures": 25, "pore-minimizing": 26,
            "excess-oil": 27, "daily-use": 28, "dullness": 29,
            "anti-acne-scarring": 30, "softening": 31, "acne": 32,
            "pore-care": 33
        };
        return concernsMapping[concern] || 0;
    };

    const concerns = [
        "anti-pollution", "tan-removal", "dryness", "deep-nourishment",
        "blackheads", "oil-control", "fine-lines", "uneven-skin-tone",
        "dark-spots", "dark-circles", "skin-tightening", "under-eye",
        "skin-inflammation", "general-care", "redness", "skin-sagging",
        "lightening", "sun-protection", "pigmentation", "blackheads-removal",
        "oily-skin", "anti-ageing", "hydration", "dull-skin",
        "uneven-texture", "irregular-textures", "pore-minimizing", "excess-oil",
        "daily-use", "dullness", "anti-acne-scarring", "softening", "acne", 
        "pore-care"
    ];

    const getAvailableConcerns = (selected) => {
        return concerns.filter(concern => concern !== selected);
    };

    return (
        <div className="form-image-container">
            <div className="blue-box">
                <Link to="/">
                    <div className="backarrow">
                        <IoMdArrowRoundBack style={{ width: 30, height: 30, marginLeft: 10, marginTop: 10 }} />
                    </div>
                </Link>
            </div>
            <div className="form-container">
                <form onSubmit={handleSubmit}>
                    <h3>Fill Below Information to know your skincare products.</h3>

                    <label>What is your skin type?</label>
                    <select name="skinType" onChange={handleChange} required>
                        <option value="">Select</option>
                        <option value="dry">Dry</option>
                        <option value="oily">Oily</option>
                        <option value="combination">Combination</option>
                        <option value="normal">Normal</option>
                    </select>

                    <label>Skin concern 1?</label>
                    <select name="concern1" onChange={handleChange} required>
                        <option value="">Select</option>
                        {concerns.map(concern => (
                            <option
                                key={concern}
                                value={concern}
                                disabled={formData.concern2 === concern || formData.concern3 === concern}
                            >
                                {concern.replace(/-/g, " ").replace(/\b\w/g, char => char.toUpperCase())}
                            </option>
                        ))}
                    </select>

                    <label>Skin concern 2?</label>
                    <select name="concern2" onChange={handleChange} required>
                        <option value="">Select</option>
                        {getAvailableConcerns(formData.concern1).map(concern => (
                            <option key={concern} value={concern} disabled={formData.concern3 === concern}>
                                {concern.replace(/-/g, " ").replace(/\b\w/g, char => char.toUpperCase())}
                            </option>
                        ))}
                    </select>

                    <label>Skin concern 3?</label>
                    <select name="concern3" onChange={handleChange} required>
                        <option value="">Select</option>
                        {getAvailableConcerns(formData.concern1)
                            .filter(concern => concern !== formData.concern2)
                            .map(concern => (
                                <option key={concern} value={concern}>
                                    {concern.replace(/-/g, " ").replace(/\b\w/g, char => char.toUpperCase())}
                                </option>
                            ))}
                    </select>

                    <button className="submit-button" type="submit">Submit</button>
                </form>
            </div>

            <div className="imager-x">
                <div>
                    <img className="image-6" src={image2} alt="" />
                </div>
                <div>
                    <img className="image-5" src={image1} alt="" />
                </div>
            </div>
        </div>
    );
};

export default Formpage;
