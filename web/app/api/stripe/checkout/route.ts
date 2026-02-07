import { auth } from '@clerk/nextjs/server'
import { NextRequest, NextResponse } from 'next/server'
import Stripe from 'stripe'

// Only initialize Stripe if key is available (prevents build errors)
const stripe = process.env.STRIPE_SECRET_KEY
  ? new Stripe(process.env.STRIPE_SECRET_KEY, { apiVersion: '2026-01-28.clover' })
  : null

// Price IDs from your Stripe dashboard
const PRICE_IDS: Record<string, { monthly: string; yearly: string }> = {
  standard: {
    monthly: process.env.STRIPE_STANDARD_MONTHLY_PRICE_ID || 'price_standard_monthly',
    yearly: process.env.STRIPE_STANDARD_YEARLY_PRICE_ID || 'price_standard_yearly',
  },
  enterprise: {
    monthly: process.env.STRIPE_ENTERPRISE_MONTHLY_PRICE_ID || 'price_enterprise_monthly',
    yearly: process.env.STRIPE_ENTERPRISE_YEARLY_PRICE_ID || 'price_enterprise_yearly',
  },
}

export async function POST(request: NextRequest) {
  try {
    if (!stripe) {
      return NextResponse.json({ error: 'Stripe not configured' }, { status: 503 })
    }

    const { userId } = await auth()

    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
    }

    const body = await request.json()
    const { plan, billingPeriod } = body

    if (!plan || !billingPeriod) {
      return NextResponse.json({ error: 'Missing plan or billing period' }, { status: 400 })
    }

    const priceId = PRICE_IDS[plan]?.[billingPeriod as 'monthly' | 'yearly']

    if (!priceId) {
      return NextResponse.json({ error: 'Invalid plan' }, { status: 400 })
    }

    // Create Stripe checkout session
    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      payment_method_types: ['card'],
      line_items: [
        {
          price: priceId,
          quantity: 1,
        },
      ],
      success_url: `${process.env.NEXT_PUBLIC_APP_URL}/dashboard?success=true`,
      cancel_url: `${process.env.NEXT_PUBLIC_APP_URL}/pricing?canceled=true`,
      metadata: {
        userId,
        plan,
        billingPeriod,
      },
      subscription_data: {
        metadata: {
          userId,
          plan,
        },
      },
    })

    return NextResponse.json({ url: session.url })
  } catch (error: any) {
    console.error('Stripe checkout error:', error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
