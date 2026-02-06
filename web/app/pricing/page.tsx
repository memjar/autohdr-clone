'use client';

import React, { useState } from 'react';
import Link from 'next/link';

// Lucide icons inline (to avoid dependency issues)
const Check = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <polyline points="20 6 9 17 4 12" />
  </svg>
);

const ArrowLeft = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

interface PricingTier {
  id: string;
  name: string;
  description: string;
  priceMonthly: number;
  priceYearly: number;
  photos: number;
  highlighted: boolean;
  badge?: string;
  features: { text: string; included: boolean }[];
  savings?: string;
}

const pricingTiers: PricingTier[] = [
  {
    id: 'free',
    name: 'Free',
    description: 'Get started with basic editing',
    priceMonthly: 0,
    priceYearly: 0,
    photos: 10,
    highlighted: false,
    features: [
      { text: '10 photos per month', included: true },
      { text: 'Basic HDR Editing', included: true },
      { text: 'Standard Sliders', included: false },
      { text: 'Object Removal', included: false },
      { text: 'Grass Greening', included: false },
      { text: 'Unused credits roll over', included: false },
      { text: 'Priority Support', included: false },
      { text: 'Automation Features', included: false }
    ]
  },
  {
    id: 'standard',
    name: 'Standard',
    description: 'Most popular for photographers',
    priceMonthly: 265,
    priceYearly: 225.25,
    photos: 500,
    highlighted: true,
    badge: 'Most popular',
    savings: '20% off yearly',
    features: [
      { text: '500 photos / month', included: true },
      { text: 'All HDR Features', included: true },
      { text: 'All Editing Sliders', included: true },
      { text: 'Object Removal (AI)', included: true },
      { text: 'Grass Greening', included: true },
      { text: 'Unused credits roll over', included: true },
      { text: 'Email Support', included: true },
      { text: 'Dropbox Automation', included: false }
    ]
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    description: 'For large-scale operations',
    priceMonthly: 2250,
    priceYearly: 2025,
    photos: 5000,
    highlighted: false,
    savings: '20% off yearly',
    features: [
      { text: '5,000 photos / month', included: true },
      { text: 'All Standard Features', included: true },
      { text: 'Auto TV Blackout', included: true },
      { text: 'Auto Add Fire', included: true },
      { text: 'Walkthrough Re-ordering', included: true },
      { text: 'Unused credits roll over', included: true },
      { text: 'Dedicated Slack Channel', included: true },
      { text: 'Priority Support 24/7', included: true }
    ]
  }
];

export default function PricingPage() {
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'yearly'>('monthly');

  const getPrice = (tier: PricingTier) => {
    return billingPeriod === 'monthly' ? tier.priceMonthly : tier.priceYearly;
  };

  const handleSubscribe = (tierId: string) => {
    // TODO: Integrate with Stripe
    console.log(`Subscribe to ${tierId}`);
    alert(`Subscribing to ${tierId} plan - Stripe integration coming soon!`);
  };

  return (
    <div className="min-h-screen bg-black text-white py-8 px-4">
      {/* Back Button */}
      <div className="max-w-7xl mx-auto mb-8">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-gray-400 hover:text-cyan-400 transition"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Editor
        </Link>
      </div>

      {/* Header */}
      <div className="max-w-7xl mx-auto text-center mb-16">
        <h1 className="text-5xl md:text-6xl font-bold mb-4">Pricing Plans</h1>
        <p className="text-xl text-gray-300 mb-8">Only Pay for What You Download</p>

        {/* Billing Toggle */}
        <div className="flex items-center justify-center gap-4 mb-12">
          <button
            onClick={() => setBillingPeriod('monthly')}
            className={`px-6 py-2 rounded-full font-semibold transition ${
              billingPeriod === 'monthly'
                ? 'bg-cyan-400 text-black'
                : 'bg-gray-800 text-white hover:bg-gray-700'
            }`}
          >
            Monthly
          </button>
          <button
            onClick={() => setBillingPeriod('yearly')}
            className={`px-6 py-2 rounded-full font-semibold transition relative ${
              billingPeriod === 'yearly'
                ? 'bg-cyan-400 text-black'
                : 'bg-gray-800 text-white hover:bg-gray-700'
            }`}
          >
            Yearly
            <span className="absolute -top-8 -right-2 text-cyan-400 text-sm font-bold whitespace-nowrap">
              up to 20% off
            </span>
          </button>
        </div>

        <div className="flex items-center justify-center gap-4">
          <p className="text-gray-400">Add Credits Anytime</p>
          <button className="px-6 py-2 border border-gray-600 rounded-lg hover:border-cyan-400 transition">
            Top Up
          </button>
        </div>
      </div>

      {/* Pricing Cards */}
      <div className="max-w-7xl mx-auto grid md:grid-cols-3 gap-8">
        {pricingTiers.map((tier) => (
          <div
            key={tier.id}
            className={`rounded-xl border transition transform hover:scale-105 ${
              tier.highlighted
                ? 'border-cyan-400 bg-gradient-to-br from-gray-900 to-black shadow-lg shadow-cyan-400/20 md:scale-105'
                : 'border-gray-800 bg-gray-950 hover:border-gray-700'
            } p-8 relative`}
          >
            {/* Badge */}
            {tier.badge && (
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-cyan-400 text-black px-4 py-1 rounded-full text-sm font-bold">
                {tier.badge}
              </div>
            )}

            {/* Tier Name */}
            <h3 className="text-3xl font-bold mb-2">{tier.name}</h3>
            <p className="text-gray-400 text-sm mb-6">{tier.description}</p>

            {/* Pricing */}
            <div className="mb-8">
              <div className="flex items-baseline gap-2 mb-4">
                <span className="text-5xl font-bold">${getPrice(tier).toFixed(2)}</span>
                <span className="text-gray-400">/ month</span>
              </div>
              <p className="text-gray-500 text-sm">
                {tier.photos.toLocaleString()} photos / month • ${tier.photos > 0 ? (getPrice(tier) / tier.photos).toFixed(2) : '0.00'}/photo
              </p>
              {billingPeriod === 'yearly' && tier.savings && (
                <p className="text-cyan-400 text-sm mt-2 font-semibold">
                  {tier.savings}
                </p>
              )}
            </div>

            {/* CTA Button */}
            <button
              onClick={() => handleSubscribe(tier.id)}
              className={`w-full py-3 rounded-lg font-semibold mb-8 transition ${
                tier.highlighted
                  ? 'bg-cyan-400 text-black hover:bg-cyan-300'
                  : 'bg-gray-800 text-white hover:bg-gray-700'
              }`}
            >
              {tier.id === 'free' ? 'Activate' : 'Get Started'}
            </button>

            {/* Features */}
            <div className="space-y-4">
              {tier.features.map((feature, idx) => (
                <div key={idx} className="flex items-start gap-3">
                  {feature.included ? (
                    <Check className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
                  ) : (
                    <div className="w-5 h-5 rounded border border-gray-700 flex-shrink-0 mt-0.5" />
                  )}
                  <span className={feature.included ? 'text-white' : 'text-gray-500'}>
                    {feature.text}
                  </span>
                </div>
              ))}
            </div>

            {/* Slider for paid tiers */}
            {(tier.id === 'standard' || tier.id === 'enterprise') && (
              <div className="mt-8 pt-8 border-t border-gray-800">
                <p className="text-gray-400 text-sm mb-4">Slide for more photos</p>
                <input
                  type="range"
                  min={tier.photos}
                  max={tier.photos * 4}
                  defaultValue={tier.photos}
                  className="w-full cursor-pointer accent-cyan-400"
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* FAQ Section */}
      <div className="max-w-4xl mx-auto mt-20">
        <h2 className="text-3xl font-bold text-center mb-12">Frequently Asked Questions</h2>
        <div className="space-y-6">
          {[
            {
              q: 'Do credits roll over?',
              a: 'Yes! Unused credits roll over to the next month indefinitely on paid plans.'
            },
            {
              q: 'Can I cancel anytime?',
              a: 'Absolutely. Cancel your subscription at any time with no penalties.'
            },
            {
              q: 'What payment methods do you accept?',
              a: 'We accept all major credit cards, debit cards, and PayPal through Stripe.'
            },
            {
              q: 'Is there a refund policy?',
              a: 'Yes, we offer a 30-day money-back guarantee on all paid plans.'
            }
          ].map((faq, idx) => (
            <div key={idx} className="bg-gray-950 border border-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-2">{faq.q}</h3>
              <p className="text-gray-400">{faq.a}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
